#pragma once
#include <windows.h>
#include <bcrypt.h>
#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include "locked_identity.h"
static std::string quote(const std::string & value) {
    std::ostringstream s; s << '"';
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') s << '\\' << char(c);
        else if (c < 32) s << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
        else s << char(c);
    }
    s << '"'; return s.str();
}
static std::string hash_file(const std::string & path) {
    BCRYPT_ALG_HANDLE algorithm{}; BCRYPT_HASH_HANDLE hash{};
    std::ifstream input(path, std::ios::binary); if (!input) throw std::runtime_error("identity file unavailable");
    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
        throw std::runtime_error("SHA256 provider unavailable");
    try {
        if (BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) < 0) throw std::runtime_error("SHA256 creation failed");
        std::array<unsigned char, 65536> buffer{}; std::array<unsigned char, 32> digest{};
        while (input) {
            input.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
            if (input.gcount() && BCryptHashData(hash, buffer.data(), ULONG(input.gcount()), 0) < 0)
                throw std::runtime_error("SHA256 update failed");
        }
        if (!input.eof()) throw std::runtime_error("identity read failed");
        if (BCryptFinishHash(hash, digest.data(), digest.size(), 0) < 0) throw std::runtime_error("SHA256 finish failed");
        BCryptDestroyHash(hash); hash = nullptr; BCryptCloseAlgorithmProvider(algorithm, 0); algorithm = nullptr;
        std::ostringstream s; s << std::hex << std::setfill('0'); for (auto x : digest) s << std::setw(2) << unsigned(x);
        return s.str();
    } catch (...) { if (hash) BCryptDestroyHash(hash); if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0); throw; }
}
static std::string verify_modules() {
    std::ostringstream out; out << '['; bool first = true;
    for (const auto & item : locked::modules) {
        HMODULE mod = GetModuleHandleA(item.name);
        char loaded[32768]{};
        if (!mod || !GetModuleFileNameA(mod, loaded, sizeof(loaded))) throw std::runtime_error("locked module not loaded");
        if (_stricmp(loaded, item.path)) throw std::runtime_error(std::string("loaded module path differs: ") + item.name);
        const std::string digest = hash_file(loaded);
        if (digest != item.sha) throw std::runtime_error(std::string("loaded module full SHA differs: ") + item.name);
        if (!first) out << ','; first=false;
        out << "{\"name\":" << quote(item.name) << ",\"path\":" << quote(loaded) << ",\"sha256\":" << quote(digest) << '}';
    }
    out << ']'; return out.str();
}
static std::string uuid_string(const cudaUUID_t & uuid) {
    std::ostringstream s; s << "GPU-" << std::hex << std::setfill('0');
    for (unsigned i=0; i<16; ++i) {
        if(i==4 || i==6 || i==8 || i==10) s << '-';
        s << std::setw(2) << unsigned(static_cast<unsigned char>(uuid.bytes[i]));
    }
    return s.str();
}
