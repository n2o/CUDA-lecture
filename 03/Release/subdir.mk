################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../main.cu 

OBJS += \
./main.o 

CU_DEPS += \
./main.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/Developer/NVIDIA/CUDA-6.5/bin/nvcc -O3 -gencode arch=compute_20,code=sm_20 --target-cpu-architecture x86 -m32 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	/Developer/NVIDIA/CUDA-6.5/bin/nvcc -O3 --compile --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 --target-cpu-architecture x86 -m32  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


