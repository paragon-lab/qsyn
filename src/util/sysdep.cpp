/****************************************************************************
  PackageName  [ util ]
  Synopsis     [ Define wrapper to system-dependent functions ]
  Author       [ Mu-Te (Joshua) Lau ]
  Copyright    [ Copyright(c) 2026 PARAG@N Lab, CS, Northwestern U, IL, USA ]
****************************************************************************/

#include "util/sysdep.hpp"

#include <limits.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <thread>

#include "spdlog/spdlog.h"
#ifdef __linux__
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

#ifndef _WIN32
#include <signal.h>
#include <sys/wait.h>
#endif

bool stop_requested();

namespace dvlab {

namespace utils {

namespace {

int wait_for_child(pid_t pid) {
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            return -1;
        }
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return 1;
}

void terminate_child_process_group(pid_t pid) {
    kill(-pid, SIGINT);
    for (int attempt = 0; attempt < 50; ++attempt) {
        int status        = 0;
        auto const waited = waitpid(pid, &status, WNOHANG);
        if (waited == pid) {
            return;
        }
        if (waited < 0 && errno != EINTR) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
    kill(-pid, SIGKILL);
    while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {
    }
}

}  // namespace

bool python_package_exists(std::string_view package_name) {
    if (!is_uv_available()) {
        spdlog::error("`uv` is required to for this command. Please install `uv` first.");
        return false;
    }
    return run_shell_command_interruptible(
               fmt::format("uv pip show {} > /dev/null 2>&1", package_name)) == 0;
}

bool pdflatex_exists() {
    return run_shell_command_interruptible("pdflatex --version > /dev/null 2>&1") == 0;
}

std::filesystem::path get_qsyn_executable_dir() {
#ifdef __linux__
    std::array<char, PATH_MAX> path{};
    ssize_t len = readlink("/proc/self/exe", path.data(), path.size() - 1);
    if (len == -1) {
        return {};
    }
    path[static_cast<size_t>(len)] = '\0';
    return std::filesystem::path(path.data()).parent_path();
#elif defined(__APPLE__)
    std::array<char, PATH_MAX> path{};
    uint32_t size = static_cast<uint32_t>(path.size());
    if (_NSGetExecutablePath(path.data(), &size) != 0) {
        return {};
    }
    return std::filesystem::path(path.data()).parent_path();
#elif defined(_WIN32)
    std::array<char, MAX_PATH> path{};
    if (GetModuleFileNameA(nullptr, path.data(), static_cast<DWORD>(path.size())) == 0) {
        return {};
    }
    return std::filesystem::path(path.data()).parent_path();
#else
    return {};
#endif
}

std::optional<std::filesystem::path> get_qsyn_config_dir() {
    auto home_dir = get_home_directory();
    if (!home_dir) {
        return std::nullopt;
    }
    return home_dir.value() + "/.config/qsyn/";
}

std::filesystem::path get_qsyn_project_dir() {
    if (char const* override_dir = std::getenv("QSYN_PROJECT")) {
        auto const path = std::filesystem::path(override_dir);
        if (std::filesystem::is_directory(path)) {
            return path;
        }
    }

    auto dir = get_qsyn_executable_dir();
    while (!dir.empty() && dir != dir.root_path()) {
        if (std::filesystem::exists(dir / "pyproject.toml")) {
            return dir;
        }
        dir = dir.parent_path();
    }

    return get_qsyn_executable_dir();
}

bool is_uv_available() {
    return run_shell_command_interruptible("uv --version > /dev/null 2>&1") == 0;
}

int run_shell_command_interruptible(std::string const& shell_command) {
#ifndef _WIN32
    pid_t const pid = fork();
    if (pid < 0) {
        std::perror("fork");
        return -1;
    }
    if (pid == 0) {
        setpgid(0, 0);
        execl("/bin/sh", "sh", "-c", shell_command.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }

    if (setpgid(pid, pid) != 0 && errno != EACCES) {
        std::perror("setpgid");
    }

    int status = 0;
    while (true) {
        auto const waited = waitpid(pid, &status, WNOHANG);
        if (waited == pid) {
            if (WIFEXITED(status)) {
                return WEXITSTATUS(status);
            }
            if (WIFSIGNALED(status)) {
                return 128 + WTERMSIG(status);
            }
            return 1;
        }
        if (waited < 0 && errno != EINTR) {
            std::perror("waitpid");
            kill(-pid, SIGKILL);
            while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {
            }
            return -1;
        }

        if (stop_requested()) {
            terminate_child_process_group(pid);
            auto const code = wait_for_child(pid);
            if (code == 128 + SIGINT || code == 130) {
                return 130;
            }
            return code >= 0 ? code : 130;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
#else
    return system(shell_command.c_str());
#endif
}

int uv_run_script(std::string_view script_path, std::vector<std::string> args) {
    if (!is_uv_available()) {
        spdlog::error(
            "`uv` is required to for this command. Please install `uv` first."
            "See `https://docs.astral.sh/uv/getting-started/installation/` for installation instructions.");
        return 1;
    }

    auto const project_dir = get_qsyn_project_dir();

    if (!std::filesystem::exists(project_dir / ".venv")) {
        spdlog::warn("No uv venv found. A new one will be created...");
        // NOTE: `uv run` will try to create a venv if it doesn't exist.
        // We don't need to create it manually. The warning is just to
        // inform the user.
    }

    auto const cmd = fmt::format(
        "uv run --project {} {} {}",
        project_dir.string(),
        script_path,
        fmt::join(args, " "));
    return run_shell_command_interruptible(cmd);
}
}  // namespace utils

}  // namespace dvlab
