#ifndef LOGREG_UTIL_TIMER_H_
#define LOGREG_UTIL_TIMER_H_

#include <chrono>

class Timer {
public:
  void Start() { 
    start_ = Clock::now(); 
  }
  double ElapseMilliSeconds() {
    Clock::time_point now = Clock::now();
    return std::chrono::duration<double, std::milli>(now - start_).count();
  }
  double ElapseSeconds() { 
    return ElapseMilliSeconds() / 1000.0; 
  }

private:
  using Clock = std::chrono::high_resolution_clock;
  Clock::time_point start_;
};

#endif  // LOGREG_UTIL_TIMER_H_
