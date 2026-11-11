CXX = g++
CXXFLAGS = -std=c++17 -O2 -pthread -Wall -Wextra -Werror
.PHONY: test clean
test: test_expert_fill
	./test_expert_fill
test_expert_fill: test_expert_fill.cpp expert_fill.h
	$(CXX) $(CXXFLAGS) test_expert_fill.cpp -o $@
clean:
	rm -f test_expert_fill
