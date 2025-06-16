docker_build('rdsea/preprocessing', 'src', 
    dockerfile="src/preprocessing/Dockerfile",
    only=["preprocessing", "util"]
)

docker_build('rdsea/ensemble', 'src', 
    dockerfile="src/ensemble/Dockerfile",
    only=["ensemble", "util"]
)

docker_build('rdsea/inference:cpu', 'src', 
    dockerfile="src/inference/Dockerfile.cpu", 
    only=["inference", "util"]
    )

k8s_yaml('src/deployment/preprocessing.yml')
k8s_yaml('src/deployment/ensemble.yml')
k8s_yaml('src/deployment/MobileNetV2.yml')
k8s_yaml('src/deployment/EfficientNetB0.yml')

k8s_yaml("src/deployment/rabbit_operator.yml")

k8s_yaml("src/deployment/rabbitmq_cluster.yml")

k8s_resource('preprocessing', port_forwards=5010)
