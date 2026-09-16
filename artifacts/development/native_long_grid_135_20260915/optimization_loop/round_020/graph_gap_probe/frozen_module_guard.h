#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>

// Backend-injected pinning policy. The host tests supply a fake API and never
// load native or CUDA libraries. Production verifies the actual loaded module.
template<class Api> class PinnedFrozenModules {
public:
    using Handle = typename Api::Handle;
    explicit PinnedFrozenModules(Api &api) : api_(api) {}
    PinnedFrozenModules(const PinnedFrozenModules &) = delete;
    PinnedFrozenModules &operator=(const PinnedFrozenModules &) = delete;
    ~PinnedFrozenModules() noexcept {
        for (auto it = handles_.rbegin(); it != handles_.rend(); ++it) api_.release(*it);
    }
    template<class Identity, std::size_t N> void preload(const Identity (&identities)[N]) {
        static_assert(N > 0, "frozen module set must be nonempty");
        if (attempted_) throw std::runtime_error("frozen module preload may execute only once");
        attempted_ = true;
        handles_.reserve(N); // allocation failure occurs before any DLL is loaded
        for (const auto &identity : identities) {
            api_.require_absolute_path(identity.path);
            Handle handle = api_.load_absolute(identity.path);
            if (!handle) throw std::runtime_error("failed to preload a frozen native DLL");
            handles_.push_back(handle); // reserved capacity; pin even when verification throws
            api_.verify_handle(handle, identity.path, identity.hash);
        }
    }
    std::size_t pinned_count() const noexcept { return handles_.size(); }
private:
    Api &api_;
    std::vector<Handle> handles_;
    bool attempted_ = false;
};
