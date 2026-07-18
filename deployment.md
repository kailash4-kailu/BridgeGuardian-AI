# BridgeGuardian AI Cloud Deployment Guide

This document describes instructions for deploying the Dockerized BridgeGuardian AI application to various cloud platforms.

---

## 🏗️ 1. Deploying to Cloud Virtual Machines
*(Applicable to: AWS EC2, Azure VM, Google Cloud VM, DigitalOcean Droplets, Hostinger VPS)*

All standard Linux VMs can deploy the application using Docker and Docker Compose.

### Step 1: Provision and Connect to your Instance
1. Spin up a VM with **Ubuntu Server 22.04 LTS** or **24.04 LTS** (minimum 2 vCPUs, 4GB RAM is recommended to run models and processing pipelines smoothly).
2. Ensure you associate an Elastic IP (AWS) or static public IP address.
3. Configure your Security Group/Firewall rules to expose the following ports:
   - **Port 22** (SSH access)
   - **Port 80** (HTTP web traffic served by Nginx)
   - **Port 443** (HTTPS secure web traffic - optional)
4. Connect to your instance via SSH:
   ```bash
   ssh -i /path/to/key.pem ubuntu@your-vm-public-ip
   ```

### Step 2: Install Docker and Docker Compose
Run the following script to install Docker:
```bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2
sudo systemctl enable docker
sudo systemctl start docker

# Add your user to the docker group to avoid running with sudo
sudo usermod -aG docker $USER
newgrp docker
```

### Step 3: Clone the Repository and Build
1. Clone the project code:
   ```bash
   git clone <your-repository-url> bridge-guardian
   cd bridge-guardian
   ```
2. Create and configure your environment variables:
   ```bash
   cp .env.example .env
   nano .env
   ```
   *Make sure you set `APP_ENV=production` and `SECRET_KEY` to a unique, random string.*
3. Launch with Docker Compose:
   ```bash
   docker compose up --build -d
   ```
4. Verify the containers are running:
   ```bash
   docker compose ps
   ```

---

## ☁️ 2. Deploying to Railway
Railway is a modern PaaS that supports zero-configuration Docker Compose deployments.

### Option A: Using the CLI
1. Install the Railway CLI:
   ```bash
   npm i -g @railway/cli
   ```
2. Login and initialize a new project in the directory:
   ```bash
   railway login
   railway init
   ```
3. Deploy the project (Railway automatically reads `docker-compose.yml` and spins up the frontend and backend services):
   ```bash
   railway up
   ```

### Option B: GitHub Integration
1. Push your repository to GitHub.
2. In the Railway dashboard, select **New Project** -> **Deploy from GitHub repo**.
3. Select your repository.
4. Railway will analyze your `docker-compose.yml` and deploy both services automatically.
5. In the settings of the `frontend` service, click **Generate Domain** to get a public URL.

---

## 🛢️ 3. Production Databases (Scaling past SQLite)
By default, the application runs on SQLite (`bridgeguardian.db`). While this is perfect for demos and small deployments, scaling to a production-grade database like PostgreSQL is simple:

1. Provision a managed PostgreSQL instance (e.g., AWS RDS, Azure Database for PostgreSQL, Supabase).
2. Retrieve the PostgreSQL connection string.
3. Update the `DATABASE_URL` in `.env`:
   ```env
   DATABASE_URL=postgresql://db_user:db_password@db_host:5432/db_name
   ```
4. Rebuild/Restart the containers. SQLAlchemy will automatically create all tables on the remote PostgreSQL database at startup:
   ```bash
   docker compose down
   docker compose up -d
   ```

---

## 🛡️ 4. Enabling SSL (HTTPS) with Let's Encrypt
To secure your deployment with HTTPS:

1. Point your domain (e.g., `bridge.yourdomain.com`) to the VM's public IP address.
2. Install Certbot on the host machine:
   ```bash
   sudo apt install -y certbot python3-certbot-nginx
   ```
3. Stop the frontend Docker container so port 80 is freed:
   ```bash
   docker compose stop frontend
   ```
4. Run Certbot to generate certificates:
   ```bash
   sudo certbot certonly --standalone -d bridge.yourdomain.com
   ```
5. Modify `nginx.conf` (or map certs inside the frontend container volumes) to enable SSL. Alternatively, use a reverse proxy on the host (like Nginx or Caddy) to handle SSL offloading and forward traffic to Docker.
