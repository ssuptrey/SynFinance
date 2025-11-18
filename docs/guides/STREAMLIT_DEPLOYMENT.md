# SynFinance - Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud (FREE)

### Step 1: Commit and Push to GitHub

```bash
# Make sure all new files are committed
git add .streamlit/config.toml
git add packages.txt
git add requirements-streamlit.txt
git add docs/planning/PRODUCTION_READINESS_ROADMAP.md

# Commit
git commit -m "Add Streamlit Cloud deployment configuration"

# Push to GitHub
git push origin main
```

### Step 2: Sign Up for Streamlit Cloud

1. Go to **https://streamlit.io/cloud**
2. Click **"Sign up"**
3. Sign up with your **GitHub account**
4. Authorize Streamlit to access your repositories

### Step 3: Deploy Your App

1. Click **"New app"** button
2. **Repository:** Select `ssuptrey/SynFinance`
3. **Branch:** `main`
4. **Main file path:** `src/app.py`
5. **Python version:** 3.9 or 3.10
6. **Requirements file:** Leave default (`requirements.txt`) OR change to `requirements-streamlit.txt` if you get memory errors
7. Click **"Deploy"**

### Step 4: Wait for Deployment (5-10 minutes)

The app will:
- Install dependencies
- Build the app
- Start the server
- Give you a public URL like: `https://synfinance-demo.streamlit.app`

### Step 5: Test Your Demo

1. Open the URL
2. Generate 100 transactions
3. Try different export formats (CSV, JSON, Excel)
4. Test fraud pattern generation
5. Make sure everything works

### Step 6: Share the URL

Your live demo is now ready! Share this URL with:
- Potential customers
- Investors
- On your landing page
- In cold emails

---

## Common Issues & Solutions

### Issue: "Out of Memory" Error

**Solution:** Use the lightweight requirements file:
1. In Streamlit Cloud dashboard, go to **Settings**
2. Change **"Requirements file"** from `requirements.txt` to `requirements-streamlit.txt`
3. Click **"Save"** and redeploy

### Issue: Import Errors

**Solution:** Make sure all your local imports work:
```bash
# Test locally first
cd E:\SynFinance
streamlit run src/app.py
```

If it works locally, it should work on Streamlit Cloud.

### Issue: App is Slow

**Solution:** This is expected on free tier. For production:
- Upgrade to Streamlit Cloud paid tier ($20-50/month)
- Or deploy to your own server (see production deployment guide)

### Issue: Need to Update App

**Solution:** Just push to GitHub:
```bash
git add .
git commit -m "Update app"
git push origin main
```

Streamlit Cloud auto-deploys on every push to main branch.

---

## Custom Domain (Optional)

To use your own domain (e.g., demo.synfinance.com):

1. Buy domain from Namecheap, GoDaddy, etc.
2. In Streamlit Cloud, go to **Settings** → **Custom domain**
3. Add your domain: `demo.synfinance.com`
4. Update your DNS with the provided CNAME record
5. Wait for DNS propagation (24-48 hours)

---

## Environment Variables / Secrets

If you need to add API keys or database credentials:

1. In Streamlit Cloud dashboard, go to **Settings** → **Secrets**
2. Add secrets in TOML format:
   ```toml
   [database]
   host = "your-db-host"
   password = "your-db-password"
   
   [api]
   stripe_key = "sk_test_xxxx"
   ```
3. Access in your app:
   ```python
   import streamlit as st
   db_password = st.secrets["database"]["password"]
   ```

---

## Monitoring

**View Logs:**
1. In Streamlit Cloud dashboard
2. Click on your app
3. Click **"Manage app"** → **"Logs"**
4. See real-time logs and errors

**View Analytics:**
- Streamlit Cloud provides basic analytics
- See number of visitors, sessions, etc.
- For detailed analytics, add Google Analytics to your app

---

## Next Steps After Deployment

✅ **You now have a live demo URL!**

1. **Test thoroughly** - Generate datasets, export, verify quality
2. **Record demo video** - Show the live URL in action
3. **Create landing page** - Add the demo link
4. **Start customer outreach** - Share the URL in emails
5. **Gather feedback** - See what people think

---

## Your Live Demo URL

After deployment, your URL will be:
**`https://[your-app-name].streamlit.app`**

Example: `https://synfinance-demo.streamlit.app`

**Write it down and share it everywhere!**

---

## Support

- **Streamlit Docs:** https://docs.streamlit.io/streamlit-cloud
- **Community Forum:** https://discuss.streamlit.io/
- **SynFinance Issues:** https://github.com/ssuptrey/SynFinance/issues

---

*Last Updated: November 18, 2025*  
*Part of: Phase 1, Day 1 - Production Readiness Roadmap*
