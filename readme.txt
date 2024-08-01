1. Download docker from https://www.docker.com/

2. Start docker desktop

3. Convert all input files to nii.gz format for predictions and copy them all to a single folder

4. Run docker commands by opening powershell or terminal:

4.1 Load docker image:
        docker load -i file_name.tar
        example:
        docker load -i kidney_segmentation.tar

4.2 Run container:
        docker run --rm --gpus all -v [path\to\patient\folder\]:/app/nnUNet_raw/Dataset020_niddk/imagesTs [docker_image_name:version]
        Note: the values in [] are to be defined by the user
        example:
        docker run --rm --gpus all -v C:\Users\ar38\Desktop\kidney_infer_nnunet\test_files:/app/nnUNet_raw/Dataset022_niddkitaly/imagesTs kidney_seg:2.0
        (--rm: optional flag that automatically deletes the container after it is finished with predictions
        --gpus all: optional flag to use available gpus on the host machine
        (if no gpu is available then remove this flag so algorithm can run on cpu)
        -v ...: mount folder containing .nii.gz files from host machine to docker container system in /app/...
        kidney_seg:1.0: docker image name with version 2.0)

5. The predicted segmentation maps will be saved in the input folder with a folder called "predictions"