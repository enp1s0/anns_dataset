CXX=g++
CXXFLAGS=-std=c++17 -Wall
CXXFLAGS+=-I../include

TARGET=ann-dataset-merge

$(TARGET):src/merge.cpp
	$(CXX) $< -o $@ $(CXXFLAGS)

clean:
	rm -f $(TARGET)
