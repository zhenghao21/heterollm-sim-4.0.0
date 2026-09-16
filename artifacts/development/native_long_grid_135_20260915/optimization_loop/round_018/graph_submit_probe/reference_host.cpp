#include "math_reference.h"
#include <iostream>
#include <stdexcept>
int main() {
    std::uint64_t checked=0;
    for (int elements : {1024,262144}) for (int nodes : {1,8,32}) {
        for (int i=0; i<elements; ++i) {
            float iter=static_cast<float>(input_numerator(i))/256.0f;
            for (int stage=1; stage<=nodes; ++stage) {
                iter=iter*0.5f;
                const float expected=reference_at(i,stage);
                if (!std::isfinite(iter) || float_bits(iter)!=float_bits(expected))
                    throw std::runtime_error("closed-form/iterative reference mismatch");
                ++checked;
            }
        }
    }
    std::cout << "{\"schema\":\"graph-submit-host-reference/v1\",\"pass\":true,\"checked_stage_elements\":" << checked << ",\"gpu_access\":false}\n";
}
