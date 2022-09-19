sudo docker run -it --rm \
                -p 5000:5000 \
                -v $(pwd):/openvino \
                --device /dev/video0:/dev/video0:mwr \
                python:3.8 bash ./openvino/run_in_docker.sh