#include "math_reference.h"
#include "frozen_module_guard.h"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Identity { const char *path; const char *hash; };
struct MockApi {
    using Handle = int;
    std::vector<int> loaded;
    std::vector<int> released;
    void require_absolute_path(const char *path) const {
        if (std::string(path).rfind("C:\\", 0) != 0) throw std::runtime_error("mock requires absolute path");
    }
    Handle load_absolute(const char *) { const int handle = int(loaded.size()) + 1; loaded.push_back(handle); return handle; }
    void verify_handle(Handle handle, const char *, const char *) const { if (handle <= 0) throw std::runtime_error("mock bad handle"); }
    void release(Handle handle) noexcept { released.push_back(handle); }
};

int main() {
    std::uint64_t checked = 0;
    for (const int elements : {262144}) for (const int nodes : {64, 256}) {
        for (int index = 0; index < elements; ++index) {
            float iterative = static_cast<float>(graph_gap_input_numerator(index)) / 256.0f;
            for (int stage = 1; stage <= nodes; ++stage) {
                iterative *= graph_gap_scale_at(stage);
                const float expected = graph_gap_reference_at(index, stage);
                if (!std::isfinite(iterative) || graph_gap_float_bits(iterative) != graph_gap_float_bits(expected)) {
                    throw std::runtime_error("closed-form/iterative reference mismatch");
                }
                ++checked;
            }
        }
    }
    MockApi api;
    const Identity identities[] = {{"C:\\one.dll", "a"}, {"C:\\two.dll", "b"}};
    {
        PinnedFrozenModules<MockApi> pinned(api);
        pinned.preload(identities);
        if (pinned.pinned_count() != 2 || api.loaded != std::vector<int>({1, 2})) throw std::runtime_error("mock preload mismatch");
    }
    if (api.released != std::vector<int>({2, 1})) throw std::runtime_error("mock release order mismatch");
    std::cout << "{\"schema\":\"graph-gap-host-reference/v1\",\"pass\":true,\"checked_stage_elements\":" << checked
              << ",\"mock_loader_pass\":true,\"gpu_access\":false}\n";
}