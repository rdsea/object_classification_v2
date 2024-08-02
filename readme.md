# Step Run

- export EXAMPLE_PATH="<path to runing example>"
- config minio:
    - start docker minio:
        ```bash
        mkdir -p ~/minio/data

        docker run \
                -p 9000:9000 \
                -p 9001:9001 \
                --name minio \
                -v ~/minio/data:/data \
                -e "MINIO_ROOT_USER=admin_user" \
                -e "MINIO_ROOT_PASSWORD=admin_pass" \
                quay.io/minio/minio server /data --console-address ":9001"
        ```
 