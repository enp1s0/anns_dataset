CXX=g++
CXXFLAGS=-std=c++17 -Wall -I../include -fopenmp

TARGET=anns-ds.test

$(TARGET):main.cpp ../include/anns_dataset.hpp
	$(CXX) $< -o $@ $(CXXFLAGS)

clean:
	rm -f $(TARGET)
