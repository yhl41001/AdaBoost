################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../AdaBoost.cpp \
../Main.cpp 

OBJS += \
./AdaBoost.o \
./Main.o 

CPP_DEPS += \
./AdaBoost.d \
./Main.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -D__GXX_EXPERIMENTAL_CXX0X__ -I/usr/local/include -O0 -g3 -Wall -c -fmessage-length=0 -std=c++0x -stdlib=libc++ -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


