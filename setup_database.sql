-- SynFinance Database Setup Script
-- Run this after connecting to PostgreSQL

-- Create user
CREATE USER synfinance_trey WITH PASSWORD 'synfinance_guccigeng77@';

-- Grant user privileges
ALTER USER synfinance_trey CREATEDB;
ALTER USER synfinance_trey WITH SUPERUSER;

-- Create databases
CREATE DATABASE synfinance OWNER synfinance_trey;
CREATE DATABASE synfinance_dev OWNER synfinance_trey;
CREATE DATABASE synfinance_test OWNER synfinance_trey;

-- Grant all privileges
GRANT ALL PRIVILEGES ON DATABASE synfinance TO synfinance_trey;
GRANT ALL PRIVILEGES ON DATABASE synfinance_dev TO synfinance_trey;
GRANT ALL PRIVILEGES ON DATABASE synfinance_test TO synfinance_trey;

-- Display success message
\echo 'Database setup completed successfully!'
\echo 'User: synfinance_trey'
\echo 'Databases: synfinance, synfinance_dev, synfinance_test'
