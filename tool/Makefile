CXX=g++
CXXFLAGS=-std=c++17 -Wall -I../include
OMPFLAGS=-fopenmp -Wall
SRCDIR=src
SRCS=$(shell find $(SRCDIR) -maxdepth 1 -name '*.cpp')
OBJDIR=objs
OBJS=$(subst $(SRCDIR),$(OBJDIR), $(SRCS))
OBJS:=$(subst .cpp,.o,$(OBJS))
HEADERS=$(shell find $(SRCDIR) -name '*.hpp' -o -name '*.h')
TARGET=anns-dataset-tool

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $+ -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -c -o $@

clean:
	rm -rf $(OBJS)
	rm -rf $(TARGET)
