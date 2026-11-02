#pragma once
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>

// Blocking FIFO. After close(), pop() drains what is left and then returns nullopt.
template <class T>
class Queue {
public:
    void push(T value) {
        {
            std::lock_guard lock(mutex_);
            items_.push_back(std::move(value));
        }
        cv_.notify_one();
    }

    void close() {
        {
            std::lock_guard lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    std::optional<T> pop() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] { return !items_.empty() || closed_; });
        if (items_.empty()) return std::nullopt;
        T value = std::move(items_.front());
        items_.pop_front();
        return value;
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<T> items_;
    bool closed_ = false;
};
