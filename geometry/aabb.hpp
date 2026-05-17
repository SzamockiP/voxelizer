#pragma once

#include <glm/glm.hpp>
#include <algorithm>
#include <limits>

namespace ic_voxel::geometry
{

    struct AABB
    {
        // Initialize with extreme values to ensure the first point sets the bounds correctly
        glm::vec3 min{ std::numeric_limits<float>::max() };
        glm::vec3 max{ std::numeric_limits<float>::lowest() };

        // Expands the bounding box to include the given point
        void expand(const glm::vec3& point)
        {
            min = glm::min(min, point);
            max = glm::max(max, point);
        }
    };

} // namespace ic_voxel::geometry
