# DiffEIC
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/DiffEIC/DiffEIC_1_2_2
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN  --model InternVL3-2B  --work-dir ./MyBenchmark/DiffEIC/DiffEIC_1_2_2
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/DiffEIC/DiffEIC_1_2_4
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN  --model InternVL3-2B  --work-dir ./MyBenchmark/DiffEIC/DiffEIC_1_2_4
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/DiffEIC/DiffEIC_1_2_8
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN  --model InternVL3-2B  --work-dir ./MyBenchmark/DiffEIC/DiffEIC_1_2_8
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/DiffEIC/DiffEIC_1_2_16
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN  --model InternVL3-2B  --work-dir ./MyBenchmark/DiffEIC/DiffEIC_1_2_16
# RDEIC
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/RDEIC/RDEIC_step2_1
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/RDEIC/RDEIC_step2_1
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/RDEIC/RDEIC_step2_2
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/RDEIC/RDEIC_step2_2
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/RDEIC/RDEIC_step2_01
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/RDEIC/RDEIC_step2_01
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/RDEIC/RDEIC_step2_05
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/RDEIC/RDEIC_step2_05
# TCM
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/TCM/TCM_00005
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/TCM/TCM_00005
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/TCM/TCM_00010
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/TCM/TCM_00010
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/TCM/TCM_00018
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/TCM/TCM_00018
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/TCM/TCM_00025
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/TCM/TCM_00025
# StableCodec
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/StableCodec/ft2
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/StableCodec/ft2
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/StableCodec/ft4
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/StableCodec/ft4
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/StableCodec/ft8
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/StableCodec/ft8
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/StableCodec/ft16
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/StableCodec/ft16
# MLIC++
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/MLICpp/q-1
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/MLICpp/q-1
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/MLICpp/q0
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/MLICpp/q0
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/MLICpp/q1
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/MLICpp/q1
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/MLICpp/q2
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/MLICpp/q2
# ELIC
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/ELIC/ELIC_0004
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/ELIC/ELIC_0004
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/ELIC/ELIC_0008
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/ELIC/ELIC_0008
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/ELIC/ELIC_0016
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/ELIC/ELIC_0016
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/ELIC/ELIC_0032
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/ELIC/ELIC_0032
# MS-ILLM
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/ILLM/ILLM_1
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/ILLM/ILLM_1
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/ILLM/ILLM_2
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/ILLM/ILLM_2
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/ILLM/ILLM_3
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/ILLM/ILLM_3
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/ILLM/ILLM_vlo2
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/ILLM/ILLM_vlo2
# VTM
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/VTM/VTM_qp40
torchrun --nproc-per-node=8 --master_port=29501 run.py --data MMBench_DEV_EN POPE   --model InternVL3-2B  --work-dir ./MyBenchmark/VTM/VTM_qp40
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/VTM/VTM_qp42
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG MME   --model InternVL3-2B  --work-dir ./MyBenchmark/VTM/VTM_qp42
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/VTM/VTM_qp45
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/VTM/VTM_qp45
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/VTM/VTM_qp49
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG MME   --model InternVL3-2B  --work-dir ./MyBenchmark/VTM/VTM_qp49
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/VTM/VTM_qp50
torchrun --nproc-per-node=8 --master_port=29501 run.py --data MMBench_DEV_EN POPE   --model InternVL3-2B  --work-dir ./MyBenchmark/VTM/VTM_qp50
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/VTM/VTM_qp53
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/VTM/VTM_qp53
# HM
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/HM/HM_qp40
torchrun --nproc-per-node=8 --master_port=29501 run.py --data MMBench_DEV_EN POPE   --model InternVL3-2B  --work-dir ./MyBenchmark/HM/HM_qp40
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/HM/HM_qp42
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG MME   --model InternVL3-2B  --work-dir ./MyBenchmark/HM/HM_qp42
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/HM/HM_qp45
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN   --model InternVL3-2B  --work-dir ./MyBenchmark/HM/HM_qp45
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/HM/HM_qp47
torchrun --nproc-per-node=8 --master_port=29501 run.py --data MMBench_DEV_EN POPE   --model InternVL3-2B  --work-dir ./MyBenchmark/HM/HM_qp47
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/HM/HM_qp49
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG MME   --model InternVL3-2B  --work-dir ./MyBenchmark/HM/HM_qp49
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/HM/HM_qp50
torchrun --nproc-per-node=8 --master_port=29501 run.py --data MMBench_DEV_EN POPE   --model InternVL3-2B  --work-dir ./MyBenchmark/HM/HM_qp50
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/HM/HM_qp51
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG MME   --model InternVL3-2B  --work-dir ./MyBenchmark/HM/HM_qp51

# HiFiC
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/HIFIC/HIFIC_low
torchrun --nproc-per-node=8 --master_port=29503 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN  --model InternVL3-2B  --work-dir ./MyBenchmark/HIFIC/HIFIC_low
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/HIFIC/HIFIC_low1
torchrun --nproc-per-node=8 --master_port=29503 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN  --model InternVL3-2B  --work-dir ./MyBenchmark/HIFIC/HIFIC_low1
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/HIFIC/HIFIC_low2
torchrun --nproc-per-node=8 --master_port=29503 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN  --model InternVL3-2B  --work-dir ./MyBenchmark/HIFIC/HIFIC_low2
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/HIFIC/HIFIC_low3
torchrun --nproc-per-node=8 --master_port=29503 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN  --model InternVL3-2B  --work-dir ./MyBenchmark/HIFIC/HIFIC_low3
# JPEG
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/JPEG/JPEG_q1
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN  --model InternVL3-2B  --work-dir ./MyBenchmark/JPEG/JPEG_q1
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/JPEG/JPEG_q6
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN  --model InternVL3-2B  --work-dir ./MyBenchmark/JPEG/JPEG_q6
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/JPEG/JPEG_q3
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN  --model InternVL3-2B  --work-dir ./MyBenchmark/JPEG/JPEG_q3
export LMUData=/data11/zhangzf2505/VLMEvalKit/CompressData/JPEG/JPEG_q5
torchrun --nproc-per-node=8 --master_port=29501 run.py --data OCRBench GQA_TestDev_Balanced SEEDBench_IMG POPE MME MMBench_DEV_EN  --model InternVL3-2B  --work-dir ./MyBenchmark/JPEG/JPEG_q5