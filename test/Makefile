CXX=g++
CXXFLAGS=-std=c++17 -I../include

TARGET=anns-ds.test

$(TARGET):main.cpp ../include/anns_dataset_loader.hpp
	$(CXX) $< -o $@ $(CXXFLAGS)
  
clean:
	rm -f $(TARGET)
