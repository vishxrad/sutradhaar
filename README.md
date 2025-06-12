# sutradhaar
how to get started with mysql

```
-- Create the database
CREATE DATABASE IF NOT EXISTS sutradhaar_db;
```
```
-- Create a user and grant privileges (replace with your own secure password)

CREATE USER 'sutradhaar_user'@'localhost' IDENTIFIED BY 'your_strong_password';
GRANT ALL PRIVILEGES ON sutradhaar_db.* TO 'sutradhaar_user'@'localhost';
FLUSH PRIVILEGES;
```