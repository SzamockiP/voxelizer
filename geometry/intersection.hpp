#pragma once

#include <glm/glm.hpp>
#include <algorithm>

namespace ic_voxel::geometry
{
	constexpr float SAT_EPSILON = 1e-5f;

	[[nodiscard]] inline bool axis_test(float p0, float p1, float p2, float box_radius)
	{
		float proj_min = std::min({ p0, p1, p2 });
		float proj_max = std::max({ p0, p1, p2 });

		return !((proj_max < -box_radius - SAT_EPSILON) || (proj_min > box_radius + SAT_EPSILON));
	}

    [[nodiscard]] inline bool check_voxel_triangle_intersect(
        const glm::vec3& box_center,
        const glm::vec3& box_half_size,
        const glm::vec3& tri_v0,
        const glm::vec3& tri_v1,
        const glm::vec3& tri_v2)
    {
        // Transition to local coordinate system (box at origin)
        glm::vec3 v0 = tri_v0 - box_center;
        glm::vec3 v1 = tri_v1 - box_center;
        glm::vec3 v2 = tri_v2 - box_center;

        // Triangle edges
        glm::vec3 e0 = v1 - v0;
        glm::vec3 e1 = v2 - v1;
        glm::vec3 e2 = v0 - v2;

        // TEST 1: AABB overlap (Standard AABB-AABB test) 
        // We calculate the min and max of the triangle vertices across all axes at once
        glm::vec3 tri_min = glm::min(glm::min(v0, v1), v2);
        glm::vec3 tri_max = glm::max(glm::max(v0, v1), v2);

        // Using glm::any and glm::greater than/less than for clean vector comparison
        if (glm::any(glm::greaterThan(tri_min, box_half_size + SAT_EPSILON)) ||
            glm::any(glm::lessThan(tri_max, -box_half_size - SAT_EPSILON)))
        {
            return false;
        }

        // TEST 2: 9 Cross products (Edge-Edge axes)
        float rad, p0, p1, p2;

        // Axes based on e0
        p0 = v0.z * e0.y - v0.y * e0.z;
        p2 = v2.z * e0.y - v2.y * e0.z;
        rad = box_half_size.y * std::abs(e0.z) + box_half_size.z * std::abs(e0.y);
        if (!axis_test(p0, p0, p2, rad)) return false;

        p0 = v0.x * e0.z - v0.z * e0.x;
        p2 = v2.x * e0.z - v2.z * e0.x;
        rad = box_half_size.x * std::abs(e0.z) + box_half_size.z * std::abs(e0.x);
        if (!axis_test(p0, p0, p2, rad)) return false;

        p0 = v0.y * e0.x - v0.x * e0.y;
        p2 = v2.y * e0.x - v2.x * e0.y;
        rad = box_half_size.x * std::abs(e0.y) + box_half_size.y * std::abs(e0.x);
        if (!axis_test(p0, p0, p2, rad)) return false;

        // Axes based on e1
        p0 = v0.z * e1.y - v0.y * e1.z;
        p1 = v1.z * e1.y - v1.y * e1.z;
        rad = box_half_size.y * std::abs(e1.z) + box_half_size.z * std::abs(e1.y);
        if (!axis_test(p0, p1, p1, rad)) return false;

        p0 = v0.x * e1.z - v0.z * e1.x;
        p1 = v1.x * e1.z - v1.z * e1.x;
        rad = box_half_size.x * std::abs(e1.z) + box_half_size.z * std::abs(e1.x);
        if (!axis_test(p0, p1, p1, rad)) return false;

        p0 = v0.y * e1.x - v0.x * e1.y;
        p1 = v1.y * e1.x - v1.x * e1.y;
        rad = box_half_size.x * std::abs(e1.y) + box_half_size.y * std::abs(e1.x);
        if (!axis_test(p0, p1, p1, rad)) return false;

        // Axes based on e2
        p0 = v0.z * e2.y - v0.y * e2.z;
        p1 = v1.z * e2.y - v1.z * e2.y;
        rad = box_half_size.y * std::abs(e2.z) + box_half_size.z * std::abs(e2.y);
        if (!axis_test(p0, p1, p0, rad)) return false;

        p0 = v0.x * e2.z - v0.z * e2.x;
        p1 = v1.x * e2.z - v1.z * e2.x;
        rad = box_half_size.x * std::abs(e2.z) + box_half_size.z * std::abs(e2.x);
        if (!axis_test(p0, p1, p0, rad)) return false;

        p0 = v0.y * e2.x - v0.x * e2.y;
        p1 = v1.y * e2.x - v1.x * e2.y;
        rad = box_half_size.x * std::abs(e2.y) + box_half_size.y * std::abs(e2.x);
        if (!axis_test(p0, p1, p0, rad)) return false;

        // TEST 3: Triangle Plane test
        glm::vec3 normal = glm::cross(e0, e1);
        float plane_dist = glm::dot(normal, v0);

        rad = glm::dot(box_half_size, glm::abs(normal));

        if (std::abs(plane_dist) > rad + SAT_EPSILON) return false;

        return true;
    }
} // namespace ic_voxel::geometry
