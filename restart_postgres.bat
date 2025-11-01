@echo off
echo Restarting PostgreSQL Service...
net stop postgresql-x64-14
timeout /t 2
net start postgresql-x64-14
echo PostgreSQL service restarted!
pause
