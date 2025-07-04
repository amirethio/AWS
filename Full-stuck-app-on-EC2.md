# 🌐 EC2 Full Stack Deployment Guide (MySQL + Node.js + React)

Deploy a full stack application on a single AWS EC2 instance (Ubuntu) with MySQL, Node.js (Express), and React.

---

## 🚀 Step 1: Create and Configure Your EC2 Instance

### ✅ Launch EC2 Instance
Choose **Ubuntu Server** as the OS.

### 🔐 Configure Security Group
In addition to default ports (`SSH: 22`, `HTTP: 80`, `HTTPS: 443`), add the following:

| Port  | Purpose                 |
|-------|-------------------------|
| 3306  | MySQL database access   |
| 3000  | React frontend          |
| 5500  | Node.js backend (or dev server) |

---

## 🛠️ Step 2: Connect to Your EC2 Instance and Update System

Open a terminal on your local machine:

```bash
ssh -i "path-to-your-key.pem" ubuntu@your-ec2-public-ip
```

Update and upgrade system packages:

```bash
sudo apt update
sudo apt upgrade -y
```

---

## 🧩 Step 3: Install and Configure MySQL

### 🔧 Install MySQL:
```bash
sudo apt install mysql-server -y
```

### 🔑 Secure MySQL and Set Password:
```bash
sudo mysql
```
Inside MySQL shell:

```sql
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'your_password';
FLUSH PRIVILEGES;
EXIT;
```

### 🔓 Login with password:
```bash
mysql -u root -p
```

### 🛠️ Create database and tables:
```sql
CREATE DATABASE your_database_name;
USE your_database_name;
-- Paste your CREATE TABLE SQL statements here
SHOW TABLES;
EXIT;
```
# check the created tables
```sql
SHOW TABLES;
EXIT;
```

### 🌍 Allow external MySQL connections:
Edit config:

```bash 
sudo nano /etc/mysql/mysql.conf.d/mysqld.cnf
```

Change bind adress to:
```
bind-address = 0.0.0.0
```

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`), then restart MySQL:

```bash
sudo systemctl restart mysql
```

### 👤 Create remote user:
```bash
mysql -u root -p
```

Inside MySQL shell:

```sql
CREATE USER 'your_user_name'@'%' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON your_database_name.* TO 'your_user_name'@'%';
FLUSH PRIVILEGES;
SELECT user, host FROM mysql.user WHERE user = 'your_user_name';
EXIT;
```

---

## ⚙️ Step 4: Install Node.js

Install Node.js v18 and npm:

```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs
```

Verify:
```bash
node -v
npm -v
```

---

## 📦 Step 5: Upload Backend or Clone from GitHub

### ✍️ Before Uploading, Edit Your DB Connection File:
```js
user: "your_user_name ",
host: "your-aws-ip",
password: "your password",
database: "your_database_name",
port: 3306,
```

### 📁 Option 1: Upload from local (delete node_modules first)
```bash
scp -i your_pem_file -r ./backend-folder-name ubuntu@your-ec2-public-ip:~
```

### 🌐 Option 2: Clone from GitHub
```bash
git clone https://github.com/your-username/your-backend-repo.git
cd your-backend-repo
```

---

## 🧪 Step 6: Install Dependencies and Run Backend with PM2

### 🧰 Install PM2 globally
```bash
sudo npm install -g pm2
```

### ▶️ Start Backend App
```bash
cd backend-folder-name
npm install
pm2 start app.js --name backend-app
pm2 save
```

---

## 🎯 Step 7: Test Backend and Upload React Frontend

### 🧪 Test API using local frontend (optional)
Update Axios config in React app:
```js
baseURL: "http://your-ec2-ip:5500/api" // adjust port  and endpoint if needed
```

### ⚙️ Build the React App
```bash
cd your-react-folder
npm run build
```

### 🚚 Upload or clone  Build Folder to EC2 
```bash
scp -i forum.pem -r ./build ubuntu@your-ec2-public-ip:~
```

---

## 🌍 Step 8: Serve React Build with `serve` and PM2

### 📦 Install `serve`
```bash
sudo npm install -g serve
```

### 🚀 Serve the build folder on port 3000:
```bash
pm2 start "serve -s /dist -l 3000" --name frontend-app    // (change dist if your build folder isn't dist)
```

---

## ✅ Final Check

Your full stack app should now be live:

```
Frontend: http://your-ec2-public-ip:3000
```

🎉 Congratulations! You’ve successfully deployed a full stack app on a single EC2 instance using MySQL, Node.js, and React.
