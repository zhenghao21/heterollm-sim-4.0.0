#include "frozen_module_guard.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <type_traits>

struct Identity { const char *path; const char *hash; };
struct FakeApi {
    using Handle = int;
    std::vector<std::string> events;
    std::vector<int> released;
    int count = 0, fail_load = 0, fail_verify = 0;
    bool null_load = false;
    void require_absolute_path(const char *path) {
        events.push_back(std::string("absolute:")+path);
        if (std::string(path).rfind("X:/",0) != 0) throw std::runtime_error("relative path");
    }
    int load_absolute(const char *path) {
        ++count; events.push_back(std::string("load:")+path);
        if (count==fail_load) throw std::runtime_error("synthetic load failure");
        return null_load ? 0 : count;
    }
    void verify_handle(int handle,const char *path,const char *hash) {
        events.push_back(std::string("verify:")+path);
        if (handle==fail_verify || std::string(hash)!="correct") throw std::runtime_error("synthetic identity mismatch");
    }
    void release(int handle) noexcept { released.push_back(handle); }
};
static void require(bool value,const char *message) { if(!value) throw std::runtime_error(message); }
static bool contains(const std::string &s,const char *p) { return s.find(p)!=std::string::npos; }
int main() {
    const Identity modules[]={{"X:/native/ggml-base.dll","correct"},{"X:/native/ggml-cpu.dll","correct"},
                              {"X:/native/ggml-cuda.dll","correct"},{"X:/native/ggml.dll","correct"}};
    int checks=0;
    { FakeApi api; { PinnedFrozenModules<FakeApi> pins(api); pins.preload(modules);
      require(pins.pinned_count()==4,"all four handles retained"); require(api.released.empty(),"premature unload");
      require(api.events[9]=="absolute:X:/native/ggml.dll" && api.events[11]=="verify:X:/native/ggml.dll","unused ggml.dll not explicitly verified"); }
      require(api.released==std::vector<int>({4,3,2,1}),"reverse release not balanced"); ++checks; }
    { FakeApi api;api.fail_load=2;bool rejected=false;
      try {PinnedFrozenModules<FakeApi> pins(api);pins.preload(modules);} catch(const std::exception &e){rejected=contains(e.what(),"load failure");}
      require(rejected && api.released==std::vector<int>({1}),"partial load cleanup");++checks; }
    { FakeApi api;api.fail_verify=2;bool rejected=false;
      try {PinnedFrozenModules<FakeApi> pins(api);pins.preload(modules);} catch(const std::exception &e){rejected=contains(e.what(),"identity mismatch");}
      require(rejected && api.released==std::vector<int>({2,1}),"bad loaded handle not released");++checks; }
    { FakeApi api;api.null_load=true;bool rejected=false;
      try {PinnedFrozenModules<FakeApi> pins(api);pins.preload(modules);} catch(const std::exception &e){rejected=contains(e.what(),"preload");}
      require(rejected && api.released.empty(),"null load silently accepted");++checks; }
    { FakeApi api;bool rejected=false;const Identity bad[]={{"relative.dll","correct"}};
      try {PinnedFrozenModules<FakeApi> pins(api);pins.preload(bad);} catch(const std::exception &e){rejected=contains(e.what(),"relative");}
      require(rejected && api.count==0,"relative load attempted");++checks; }
    { FakeApi api;bool rejected=false;{PinnedFrozenModules<FakeApi> pins(api);pins.preload(modules);
      try {pins.preload(modules);} catch(const std::exception &e){rejected=contains(e.what(),"only once");}
      require(rejected && api.count==4,"duplicate preload changed refcounts");}++checks; }
    { FakeApi api;bool rejected=false;const Identity bad[]={{"X:/native/ggml.dll","wrong-hash"}};
      try {PinnedFrozenModules<FakeApi> pins(api);pins.preload(bad);} catch(const std::exception &e){rejected=contains(e.what(),"identity mismatch");}
      require(rejected && api.released==std::vector<int>({1}),"hash mismatch accepted");++checks; }
    static_assert(!std::is_copy_constructible<PinnedFrozenModules<FakeApi>>::value,"copying pins would double-unload");++checks;
    std::cout << "{\"status\":\"passed\",\"checks\":" << checks
              << ",\"gpu_access\":false,\"native_libraries_loaded\":false,\"API\":\"fake_only\"}\n";
    return 0;
}
