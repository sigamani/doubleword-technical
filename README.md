michaelsigamani@MacBookPro ~ % ssh -p 40195 root@77.104.167.148 \
       -L 8000:localhost:8000 \
       -L 3000:localhost:3000 \
       -L 9090:localhost:9090 \
       -L 8265:localhost:8265

----------
root@ubuntu:~/workspace# docker ps
CONTAINER ID   IMAGE                                            COMMAND                  CREATED          STATUS          PORTS                                                                                            NAMES
b62544e571f5   nginx:alpine                                     "/docker-entrypoint.…"   18 minutes ago   Up 18 minutes   0.0.0.0:8080->80/tcp, [::]:8080->80/tcp                                                          nginx-lb
97047289c02e   michaelsigamani/proj-grounded-telescopes:0.1.0   "/opt/nvidia/nvidia_…"   20 minutes ago   Up 20 minutes   6006/tcp, 6379/tcp, 8265/tcp, 8888/tcp, 10001/tcp, 0.0.0.0:8001->8000/tcp, [::]:8001->8000/tcp   inference-worker-2
ca038ec62bf4   michaelsigamani/proj-grounded-telescopes:0.1.0   "/opt/nvidia/nvidia_…"   35 minutes ago   Up 35 minutes   6006/tcp, 6379/tcp, 8265/tcp, 8888/tcp, 10001/tcp, 0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp   inference-server
c0455ab80cd7   grafana/grafana:latest                           "/run.sh"                53 minutes ago   Up 53 minutes   0.0.0.0:3000->3000/tcp, [::]:3000->3000/tcp                                                      grafana
3e451eb297b0   prom/node-exporter:latest                        "/bin/node_exporter …"   53 minutes ago   Up 53 minutes   0.0.0.0:9100->9100/tcp, [::]:9100->9100/tcp                                                      node-exporter
0840316980fe   prom/prometheus:latest                           "/bin/prometheus --c…"   53 minutes ago   Up 48 minutes   0.0.0.0:9090->9090/tcp, [::]:9090->9090/tcp                                                      prometheus
27d891d88a3e   nvidia/dcgm-exporter:latest                      "/usr/local/dcgm/dcg…"   53 minutes ago   Up 53 minutes   0.0.0.0:9400->9400/tcp, [::]:9400->9400/tcp

--------

python llm_client.py --server http://10.0.2.15:8000 --health

# Send single request
python llm_client.py --server http://10.0.2.15:8000 --prompt "What is AI?"

# Send batch requests
python llm_client.py --server http://10.0.2.15:8000 --batch

# Process prompts from file
python llm_client.py --server http://10.0.2.15:8000 --file prompts.txt --batch

-------


1. SSH into your second machine                                                                                                                         

2. Install the same Docker image:                                                                                                                   
docker pull michaelsigamani/proj-grounded-telescopes:0.1.0                                                                                         
                                                                                                                                                  
3. Start the worker container:
docker run -d --name ray-worker \
  --gpus all \
  -p 8002:8000 \
  michaelsigamani/proj-grounded-telescopes:0.1.0

4. Connect to Ray cluster:
docker exec ray-worker ray start --address='77.104.167.148:6379' --redis-password='ray123'

5. Copy and run the worker script:
# Copy this to your second machine
docker exec ray-worker bash -c "cd /workspace && python ray_cluster_fixed.py worker 77.104.167.148:6379"

Current Head Node Info:
- IP: 77.104.167.148
- Ray Port: 6379
- Dashboard: http://77.104.167.148:8265
