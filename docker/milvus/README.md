# Install Milvus

We follow the Milvus instructions at [here](https://milvus.io/docs/install_standalone-docker-compose.md) to install Milvus **Standalone** Service on a c5.xlarge instance. 

## Configure Milvus

We follow the configuration instructions at [here](https://milvus.io/docs/configure-docker.md) to modify the `docker-compose.yml` file to

* Reference the `milvus.yaml` file
* Store Milvus vector DB data to `/home/ubuntu/milvus_data_retriever` to prevent running out of storage space.

```shell
volumes:
  - /home/ubuntu/denser_retriever/milvus/standalone/milvus.yaml:/milvus/configs/milvus.yaml   # Map the local path to the container path
  - ${DOCKER_VOLUME_DIRECTORY:-/home/ubuntu/milvus_data_retriever}/volumes/milvus:/var/lib/milvus
```

## Start Milvus

Follow the instructions at [here](https://milvus.io/docs/install_standalone-docker-compose.md), we run the following command to start Milvus:

```shell
cd standalone
sudo docker compose up -d
```

## User Access Authentication

We follow the instructions at [here](https://milvus.io/docs/authenticate.md) to enable user access authentication. Specifically, we set `common.security.authorizationEnabled` in `milvus.yaml` as `true` when configuring Milvus.

A `root` user (password: `Milvus`) is created along with each Milvus instance by default. We run the following cmd to change the `root`'s password. 

```python
python reset_password.py
```

## Test Milvus

To test milvus connection and operators, we run the following command from any computers

```python
python hello_milvus.py
```

##  Stop Milvus

To stop the Milvus, we go to the Milvus server and run the following command

```shell
sudo docker compose stop
```




