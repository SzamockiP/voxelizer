#include <print>
#include <string>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <optional>
#include <queue>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>

#include "geometry/aabb.hpp"
#include "geometry/intersection.hpp"
#include "math/morton.hpp"

// third_party
#include "glm/glm.hpp"
#include "CLI11.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_DISABLE_FAST_FLOAT
#include "tiny_obj_loader.h"

using namespace ic_voxel;

// =============================================================================
// Data types
// =============================================================================

struct Triangle
{
	glm::vec3 v0, v1, v2;
};

struct AppConfig
{
	std::filesystem::path input;
	std::filesystem::path output;
	std::filesystem::path temp_dir;
	std::uint32_t resolution         = 1024;
	std::size_t   max_ram_voxels     = 1'000'000'000;
	std::size_t   max_voxels_per_tri = 100'000;
};

struct LoadedModel
{
	std::vector<Triangle> triangles;
	geometry::AABB        aabb;
};

// =============================================================================
// Chunk I/O helpers
// =============================================================================

// Mutex for thread-safe console output (internal to this TU)
static std::mutex g_print_mutex;

void flush_voxel_buffer(std::vector<math::MortonCode>& buffer,
                        unsigned int chunk_id,
                        const std::filesystem::path& temp_dir)
{
	if (buffer.empty()) return;

	// Sort and deduplicate before saving to reduce file size and later merge overhead
	std::sort(buffer.begin(), buffer.end());
	buffer.erase(std::unique(buffer.begin(), buffer.end()), buffer.end());

	const std::filesystem::path path = temp_dir / std::format("temp_chunk_{}.bin", chunk_id);
	std::ofstream out(path, std::ios::binary);
	if (!out.is_open())
		throw std::runtime_error(std::format("Failed to open temp file for writing: {}", path.string()));

	out.write(reinterpret_cast<const char*>(buffer.data()), buffer.size() * sizeof(math::MortonCode));
	buffer.clear();
}

struct MergeItem
{
	math::MortonCode morton;
	int              stream_index;

	// Full ordering via spaceship — std::greater<MergeItem> works for the min-heap
	auto operator<=>(const MergeItem&) const = default;
};

void merge_and_deduplicate_chunks(unsigned int chunk_count,
                                  const std::filesystem::path& temp_dir,
                                  const std::filesystem::path& out_path)
{
	std::println("\nMerging {} temp files", chunk_count);

	std::vector<std::ifstream> streams;
	std::priority_queue<MergeItem, std::vector<MergeItem>, std::greater<MergeItem>> pq;

	for (unsigned int i = 0; i < chunk_count; ++i)
	{
		const auto chunk_path = temp_dir / std::format("temp_chunk_{}.bin", i);
		streams.emplace_back(chunk_path, std::ios::binary);
		if (!streams.back().is_open())
			throw std::runtime_error(std::format("Couldn't open file to merge: {}", chunk_path.string()));

		math::MortonCode first;
		if (streams.back().read(reinterpret_cast<char*>(&first), sizeof(math::MortonCode)))
			pq.push({ first, static_cast<int>(i) });
	}

	std::ofstream out(out_path, std::ios::binary);
	if (!out.is_open())
		throw std::runtime_error(std::format("Couldn't create output file: {}", out_path.string()));

	constexpr std::size_t MERGE_BUFFER_SIZE = 10'000'000;
	std::vector<math::MortonCode> out_buffer;
	out_buffer.reserve(MERGE_BUFFER_SIZE);

	// std::optional makes the "no previous value" intent explicit — no magic sentinel
	std::optional<math::MortonCode> last_morton;
	std::size_t final_voxels = 0;

	while (!pq.empty())
	{
		const MergeItem current = pq.top();
		pq.pop();

		if (current.morton != last_morton)
		{
			out_buffer.push_back(current.morton);
			last_morton = current.morton;
			++final_voxels;

			if (out_buffer.size() >= MERGE_BUFFER_SIZE)
			{
				out.write(reinterpret_cast<const char*>(out_buffer.data()),
				          out_buffer.size() * sizeof(math::MortonCode));
				out_buffer.clear();
			}
		}

		math::MortonCode next;
		if (streams[current.stream_index].read(reinterpret_cast<char*>(&next), sizeof(math::MortonCode)))
			pq.push({ next, current.stream_index });
	}

	if (!out_buffer.empty())
		out.write(reinterpret_cast<const char*>(out_buffer.data()),
		          out_buffer.size() * sizeof(math::MortonCode));

	for (auto& s : streams) s.close();

	for (unsigned int i = 0; i < chunk_count; ++i)
		std::filesystem::remove(temp_dir / std::format("temp_chunk_{}.bin", i));

	if (std::filesystem::is_empty(temp_dir))
		std::filesystem::remove(temp_dir);

	std::println("Files merged\n - Size: {:.2f} MB\n - Saved: {} voxels",
	             static_cast<double>(final_voxels * sizeof(math::MortonCode)) / (1024.0 * 1024.0),
	             final_voxels);
}

// =============================================================================
// Pipeline stages
// =============================================================================

// Parses CLI arguments and validates/resolves all paths.
// CLI11_PARSE may call exit() on --help or bad args — that is intentional.
AppConfig parse_args(int argc, char** argv)
{
	CLI::App app{ "ObjParser - .obj to morton code ordered voxel storage file (Multithreaded)" };

	std::string input_str, output_str, temp_str;
	AppConfig cfg;

	app.add_option("-i,--input",        input_str,            ".obj input filepath")->required()->check(CLI::ExistingFile);
	app.add_option("-o,--output",       output_str,           ".bin output filepath")->required();
	app.add_option("-t,--temp",         temp_str,             "Temp folder directory path");
	app.add_option("-r,--resolution",   cfg.resolution,       "Voxel grid resolution (default: 1024)");
	app.add_option("-m,--max-ram",      cfg.max_ram_voxels,   "Maximum total voxels in RAM before flushing (default: 1000M)");
	app.add_option("--max-voxels-tri",  cfg.max_voxels_per_tri, "Max voxels in a triangle's AABB before subdivision (default: 100000)");

	CLI11_PARSE(app, argc, argv);

	if (!output_str.ends_with(".bin"))
		output_str += ".bin";

	cfg.input  = input_str;
	cfg.output = output_str;

	if (cfg.output.has_parent_path() && !std::filesystem::exists(cfg.output.parent_path()))
		throw std::runtime_error("Output directory doesn't exist.");

	if (!temp_str.empty())
	{
		cfg.temp_dir = temp_str;
		if (!std::filesystem::is_directory(cfg.temp_dir))
			throw std::runtime_error("Provided temp path does not exist or is not a directory.");
	}
	else
	{
		cfg.temp_dir = cfg.output.has_parent_path()
		             ? cfg.output.parent_path() / "temp"
		             : std::filesystem::path("temp");

		if (!std::filesystem::exists(cfg.temp_dir))
			std::filesystem::create_directories(cfg.temp_dir);
	}

	std::println("Temp directory set to: {}", cfg.temp_dir.string());
	return cfg;
}

// Loads an .obj file, flattens all shapes into a triangle list, and computes the AABB.
LoadedModel load_obj(const AppConfig& cfg)
{
	tinyobj::ObjReaderConfig reader_cfg;
	reader_cfg.mtl_search_path = cfg.input.parent_path().string();

	std::println("\nLoading model: {}", cfg.input.string());
	tinyobj::ObjReader reader;
	if (!reader.ParseFromFile(cfg.input.string(), reader_cfg) && !reader.Error().empty())
		throw std::runtime_error(std::format("TinyObjReader: {}", reader.Error()));

	if (!reader.Warning().empty())
		std::println("TinyObjReader warning: {}", reader.Warning());

	const auto& attrib = reader.GetAttrib();
	const auto& shapes = reader.GetShapes();

	std::println("Model loaded.\n - Vertices: {}\n - Shapes: {}", attrib.vertices.size() / 3, shapes.size());
	std::println("\nFlattening geometry & computing bounding box");

	LoadedModel model;

	for (const auto& shape : shapes)
	{
		std::size_t index_offset = 0;
		for (std::size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f)
		{
			const auto face_verts = shape.mesh.num_face_vertices[f];
			if (face_verts != 3) { index_offset += face_verts; continue; }

			const auto idx0 = shape.mesh.indices[index_offset];
			const auto idx1 = shape.mesh.indices[index_offset + 1];
			const auto idx2 = shape.mesh.indices[index_offset + 2];

			Triangle tri;
			tri.v0 = { attrib.vertices[3 * idx0.vertex_index],     attrib.vertices[3 * idx0.vertex_index + 1], attrib.vertices[3 * idx0.vertex_index + 2] };
			tri.v1 = { attrib.vertices[3 * idx1.vertex_index],     attrib.vertices[3 * idx1.vertex_index + 1], attrib.vertices[3 * idx1.vertex_index + 2] };
			tri.v2 = { attrib.vertices[3 * idx2.vertex_index],     attrib.vertices[3 * idx2.vertex_index + 1], attrib.vertices[3 * idx2.vertex_index + 2] };

			model.aabb.expand(tri.v0);
			model.aabb.expand(tri.v1);
			model.aabb.expand(tri.v2);
			model.triangles.push_back(tri);
			index_offset += 3;
		}
	}

	std::println(" - Triangles: {}", model.triangles.size());
	std::println(" - AABB min: ({:.3f}, {:.3f}, {:.3f})", model.aabb.min.x, model.aabb.min.y, model.aabb.min.z);
	std::println(" - AABB max: ({:.3f}, {:.3f}, {:.3f})", model.aabb.max.x, model.aabb.max.y, model.aabb.max.z);

	return model;
}

// Recursively subdivides triangles whose AABB covers more than max_voxels_per_tri voxels.
// Splits along the longest edge to keep sub-triangles as equilateral as possible.
void subdivide_triangles(std::vector<Triangle>& triangles,
                         float inv_voxel_size,
                         std::size_t max_voxels_per_tri)
{
	std::println("\nOptimizing geometry (subdividing large triangles)...");
	const std::size_t initial_count = triangles.size();

	for (std::size_t i = 0; i < triangles.size();)
	{
		const Triangle& tri = triangles[i];

		geometry::AABB aabb;
		aabb.expand(tri.v0); aabb.expand(tri.v1); aabb.expand(tri.v2);
		const glm::vec3 sz = aabb.max - aabb.min;

		const int vx = std::max(1, static_cast<int>(std::ceil(sz.x * inv_voxel_size)));
		const int vy = std::max(1, static_cast<int>(std::ceil(sz.y * inv_voxel_size)));
		const int vz = std::max(1, static_cast<int>(std::ceil(sz.z * inv_voxel_size)));

		if (static_cast<std::size_t>(vx) * vy * vz > max_voxels_per_tri)
		{
			const float d01 = glm::distance(tri.v0, tri.v1);
			const float d12 = glm::distance(tri.v1, tri.v2);
			const float d20 = glm::distance(tri.v2, tri.v0);

			Triangle t1, t2;
			if (d01 >= d12 && d01 >= d20)      { const glm::vec3 m = (tri.v0 + tri.v1) * 0.5f; t1 = { tri.v0, m, tri.v2 }; t2 = { m, tri.v1, tri.v2 }; }
			else if (d12 >= d01 && d12 >= d20) { const glm::vec3 m = (tri.v1 + tri.v2) * 0.5f; t1 = { tri.v1, m, tri.v0 }; t2 = { m, tri.v2, tri.v0 }; }
			else                               { const glm::vec3 m = (tri.v2 + tri.v0) * 0.5f; t1 = { tri.v2, m, tri.v1 }; t2 = { m, tri.v0, tri.v1 }; }

			triangles[i] = t1;
			triangles.push_back(t2);
		}
		else
		{
			++i;
		}
	}

	std::println(" - Initial: {}  |  After subdivision: {}", initial_count, triangles.size());
}

// Multithreaded voxelization: tests each triangle against the voxel grid,
// writes morton-coded voxels to temp chunk files, then merges them.
void voxelize(const LoadedModel& model, const AppConfig& cfg)
{
	const geometry::AABB& aabb = model.aabb;

	const glm::vec3 extent       = aabb.max - aabb.min;
	const float     voxel_size   = std::max({ extent.x, extent.y, extent.z }) / static_cast<float>(cfg.resolution);
	const float     inv_vox      = 1.0f / voxel_size;
	const glm::vec3 box_half     { voxel_size * 0.5f };

	const auto world_to_voxel = [&](float pos, float global_min) -> int
	{
		return std::clamp(static_cast<int>(std::floor((pos - global_min) * inv_vox)),
		                  0, static_cast<int>(cfg.resolution) - 1);
	};

	const unsigned int num_threads = [] {
		const unsigned int n = std::thread::hardware_concurrency();
		return n > 0 ? n : 4u;
	}();
	const std::size_t max_per_thread = cfg.max_ram_voxels / num_threads;
	const std::size_t total_tris     = model.triangles.size();

	std::println("\nStarting voxelization");
	std::println(" - Threads: {}  |  Max RAM: {} voxels (~{} per thread)", num_threads, cfg.max_ram_voxels, max_per_thread);

	std::atomic<std::size_t>  current_tri{ 0 };
	std::atomic<unsigned int> chunk_counter{ 0 };
	std::atomic<std::size_t>  total_generated{ 0 };
	std::atomic<bool>         processing_done{ false };
	std::atomic<bool>         worker_failed{ false };
	std::exception_ptr        worker_exception;
	std::mutex                worker_exception_mutex;

	auto worker = [&]()
	{
		try
		{
			std::vector<math::MortonCode> local_buffer;
			local_buffer.reserve(max_per_thread);
			std::size_t local_generated = 0;

			while (true)
			{
				if (worker_failed.load(std::memory_order_relaxed)) return;

				const std::size_t idx = current_tri.fetch_add(1, std::memory_order_relaxed);
				if (idx >= total_tris) break;

				const Triangle& tri = model.triangles[idx];

				geometry::AABB tri_aabb;
				tri_aabb.expand(tri.v0); tri_aabb.expand(tri.v1); tri_aabb.expand(tri.v2);

				const int x0 = world_to_voxel(tri_aabb.min.x, aabb.min.x), x1 = world_to_voxel(tri_aabb.max.x, aabb.min.x);
				const int y0 = world_to_voxel(tri_aabb.min.y, aabb.min.y), y1 = world_to_voxel(tri_aabb.max.y, aabb.min.y);
				const int z0 = world_to_voxel(tri_aabb.min.z, aabb.min.z), z1 = world_to_voxel(tri_aabb.max.z, aabb.min.z);

				for (int z = z0; z <= z1; ++z)
				for (int y = y0; y <= y1; ++y)
				for (int x = x0; x <= x1; ++x)
				{
					const glm::vec3 center = {
						aabb.min.x + (x + 0.5f) * voxel_size,
						aabb.min.y + (y + 0.5f) * voxel_size,
						aabb.min.z + (z + 0.5f) * voxel_size
					};

					if (geometry::check_voxel_triangle_intersect(center, box_half, tri.v0, tri.v1, tri.v2))
					{
						++local_generated;
						local_buffer.push_back(math::encode_morton(x, y, z));

						if (local_buffer.size() >= max_per_thread)
						{
							const unsigned int chunk_id = chunk_counter.fetch_add(1, std::memory_order_relaxed);
							{
								std::lock_guard lock(g_print_mutex);
								std::print("\r{:<80}\r", " ");
								std::println("Thread {} flushing temp file: {}", std::this_thread::get_id(), chunk_id);
							}
							flush_voxel_buffer(local_buffer, chunk_id, cfg.temp_dir);
						}
					}
				}
			}

			if (!local_buffer.empty())
			{
				const unsigned int chunk_id = chunk_counter.fetch_add(1, std::memory_order_relaxed);
				{
					std::lock_guard lock(g_print_mutex);
					std::print("\r{:<80}\r", " ");
					std::println("Thread {} flushing final temp file: {}", std::this_thread::get_id(), chunk_id);
				}
				flush_voxel_buffer(local_buffer, chunk_id, cfg.temp_dir);
			}

			total_generated.fetch_add(local_generated, std::memory_order_relaxed);
		}
		catch (...)
		{
			worker_failed.store(true, std::memory_order_relaxed);
			std::lock_guard lock(worker_exception_mutex);
			if (!worker_exception)
				worker_exception = std::current_exception();
		}
	};

	// Launch workers
	std::vector<std::thread> threads;
	threads.reserve(num_threads);
	for (unsigned int i = 0; i < num_threads; ++i)
		threads.emplace_back(worker);

	// Dedicated progress reporting thread
	std::thread progress([&]()
	{
		while (!processing_done.load(std::memory_order_relaxed))
		{
			const std::size_t cur     = std::min(current_tri.load(std::memory_order_relaxed), total_tris);
			const double      percent = total_tris > 0 ? static_cast<double>(cur) / total_tris * 100.0 : 100.0;
			{
				std::lock_guard lock(g_print_mutex);
				std::print("\r[ Progress ] {} / {} triangles ({:.2f}%)", cur, total_tris, percent);
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
		std::lock_guard lock(g_print_mutex);
		std::print("\r[ Progress ] {} / {} triangles (100.00%)\n", total_tris, total_tris);
	});

	for (auto& t : threads) t.join();
	processing_done.store(true, std::memory_order_relaxed);
	progress.join();

	// Re-throw any worker exception in the main thread
	if (worker_exception)
		std::rethrow_exception(worker_exception);

	const unsigned int chunks = chunk_counter.load();
	if (chunks > 0)
	{
		std::println("\nFinishing");
		std::println(" - Processed: {} triangles", total_tris);
		std::println(" - Generated: {} voxels (before final deduplication)", total_generated.load());
		merge_and_deduplicate_chunks(chunks, cfg.temp_dir, cfg.output);
	}
	else
	{
		std::println("No voxels generated. Output file hasn't been created.");
	}
}

// =============================================================================
// Entry point
// =============================================================================

int main(int argc, char** argv)
{
	try
	{
		const AppConfig cfg   = parse_args(argc, argv);
		      LoadedModel model = load_obj(cfg);

		const glm::vec3 extent     = model.aabb.max - model.aabb.min;
		const float     max_dim    = std::max({ extent.x, extent.y, extent.z });
		const float     voxel_size = max_dim / static_cast<float>(cfg.resolution);

		std::println(" - Max model size: {:.3f}", max_dim);
		std::println(" - Voxel size:     {:.3f}", voxel_size);

		subdivide_triangles(model.triangles, 1.0f / voxel_size, cfg.max_voxels_per_tri);
		voxelize(model, cfg);
	}
	catch (const std::exception& e)
	{
		std::println(stderr, "\nFATAL ERROR: {}", e.what());
		return 1;
	}

	return 0;
}
