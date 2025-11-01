# pgAdmin Setup Guide for SynFinance

## Connecting to PostgreSQL as synfinance_trey

Since we created a dedicated user for SynFinance, you should connect to pgAdmin using that user instead of the `postgres` superuser.

### Steps to Add New Server in pgAdmin:

1. **Open pgAdmin**

2. **Right-click on "Servers" in the left panel**
   - Select "Register" → "Server..."

3. **General Tab:**
   - **Name:** `SynFinance Local`

4. **Connection Tab:**
   - **Host name/address:** `localhost`
   - **Port:** `5432`
   - **Maintenance database:** `synfinance`
   - **Username:** `synfinance_trey`
   - **Password:** `synfinance_guccigeng77@`
   - ✅ Check "Save password?" (optional, for convenience)

5. **Click "Save"**

### What You'll See:

After connecting, you should see:
```
Servers
└── SynFinance Local
    └── Databases (3)
        ├── synfinance
        ├── synfinance_dev
        └── synfinance_test
```

Under each database → Schemas → public → Tables, you'll see:
- customers
- merchants
- ml_features
- transactions

---

## If You Want to Reset the postgres Password (Optional)

If you need to access the `postgres` superuser account, you can reset its password:

### Method 1: Using Command Line (with temporary trust auth)

1. **Edit pg_hba.conf** (requires admin rights)
   - Location: `C:\Program Files\PostgreSQL\14\data\pg_hba.conf`
   - Change local connections to `trust`:
   ```
   # IPv4 local connections:
   host    all             all             127.0.0.1/32            trust
   # IPv6 local connections:
   host    all             all             ::1/128                 trust
   ```

2. **Restart PostgreSQL service**
   ```cmd
   net stop postgresql-x64-14
   net start postgresql-x64-14
   ```

3. **Reset postgres password**
   ```cmd
   psql -U postgres -c "ALTER USER postgres WITH PASSWORD 'your_new_password';"
   ```

4. **Revert pg_hba.conf back to secure method** (`scram-sha-256` or `md5`)

5. **Restart PostgreSQL again**

### Method 2: Using pgAdmin with synfinance_trey

If you're already connected as `synfinance_trey`, you won't have permission to change the `postgres` password (it's a superuser). But you don't need it for SynFinance development!

---

## Recommended Approach

**For SynFinance development, you only need `synfinance_trey`:**
- ✅ Can access all SynFinance databases
- ✅ Can create/modify tables in those databases
- ✅ No need for superuser privileges
- ✅ More secure (principle of least privilege)

**Only use `postgres` if you need to:**
- Create new database users
- Create new databases
- Perform PostgreSQL system administration tasks

---

## Current Working Configuration

Your application is already configured to use `synfinance_trey`:

**Config files:**
- `config/default.yaml`
- `config/development.yaml`
- `config/test.yaml`

**Database manager:**
- `src/database/db_manager.py` (default credentials set)

**All tests passing with this configuration!** ✅

---

## Troubleshooting

### Issue: pgAdmin shows "password authentication failed for user postgres"
**Solution:** Don't use the default postgres server connection. Create a new server connection for `synfinance_trey` (see steps above).

### Issue: Can't see my databases in pgAdmin
**Solution:** Make sure you connected as `synfinance_trey` and selected the correct maintenance database (`synfinance`).

### Issue: Can't create new tables
**Solution:** You shouldn't need to manually create tables - use `python scripts/init_database.py` to create them from SQLAlchemy models.

---

**Bottom Line:** For SynFinance development, forget about the `postgres` user and just use `synfinance_trey` in pgAdmin! 🎯
