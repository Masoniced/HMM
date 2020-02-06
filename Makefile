CXX = g++
CXXFLAG = -std=c++11 -Wall -O3


ifeq ($(LANG),)
CP = copy
RM = del
else
CP = cp
RM = rm -rf
endif

HMM: HMM_example.o HMM.o
	$(CXX) $(CXXFLAG) -o HMM HMM_example.o HMM.o

HMM_example.o: HMM_example.cpp HMM.h
	$(CXX) $(CXXFLAG) -c HMM_example.cpp

HMM.o: HMM.cpp HMM.h
	$(CXX) $(CXXFLAG) -c HMM.cpp

	echo make done

clean:
	$(RM) *.o HMM HMM.exe
	echo clean done