# 🌐 DNS SETUP für NOBELBRETT.DE bei IONOS

## Schritt-für-Schritt Anleitung für www.nobelbrett.de

### 1. 📍 Server IP-Adresse ermitteln
```bash
# Aktuelle Server IP anzeigen
curl -4 ifconfig.co
```

### 2. 🌐 IONOS DNS-Einstellungen

**Login bei IONOS:**
1. Melde dich bei [IONOS](https://www.ionos.de) an
2. Gehe zu "Domains & SSL"
3. Wähle "nobelbrett.de" aus
4. Klicke auf "DNS verwalten"

**DNS Records setzen:**

| Typ | Hostname | Wert | TTL |
|-----|----------|------|-----|
| A | www | `DEINE_SERVER_IP` | 3600 |
| A | @ | `DEINE_SERVER_IP` | 3600 |
| CNAME | nobelbrett.de | www.nobelbrett.de | 3600 |

### 3. 🔒 SSL-Zertifikat (Let's Encrypt)

**Installation:**
```bash
# Certbot installieren
sudo apt update
sudo apt install certbot python3-certbot-nginx

# SSL-Zertifikat erstellen
sudo certbot --nginx -d www.nobelbrett.de -d nobelbrett.de
```

### 4. 🌐 Nginx Reverse Proxy Setup

**Nginx installieren (falls nicht vorhanden):**
```bash
sudo apt install nginx
```

**Domain-Konfiguration:**
```bash
# Nginx config erstellen
sudo python3 /root/tradino/deploy/nobelbrett_dashboard.py --setup-nginx

# Nginx testen und neustarten
sudo nginx -t
sudo systemctl restart nginx
```

### 5. 🔧 Auto-Start Service

**Systemd Service erstellen:**
```bash
sudo python3 /root/tradino/deploy/nobelbrett_dashboard.py --setup-service
sudo systemctl start nobelbrett-dashboard
sudo systemctl status nobelbrett-dashboard
```

### 6. ✅ DNS Propagation prüfen

**Tools zum Testen:**
```bash
# DNS lookup
nslookup www.nobelbrett.de

# Online Tools:
# https://www.whatsmydns.net/
# https://dnschecker.org/
```

### 7. 🚀 Dashboard starten

```bash
# Direkter Start für Testing
cd /root/tradino
python3 deploy/nobelbrett_dashboard.py --production

# Oder via Service
sudo systemctl start nobelbrett-dashboard
```

### 8. 🌐 Zugriff testen

**URLs testen:**
- https://www.nobelbrett.de
- https://nobelbrett.de (sollte auf www umleiten)
- https://www.nobelbrett.de/health (Health Check)

### 🔧 Troubleshooting

**DNS nicht erreichbar:**
```bash
# Server-IP prüfen
ip addr show

# Firewall prüfen
sudo ufw status
sudo ufw allow 80
sudo ufw allow 443
```

**SSL-Probleme:**
```bash
# Zertifikat erneuern
sudo certbot renew --dry-run

# Nginx Logs prüfen
sudo tail -f /var/log/nginx/error.log
```

**Dashboard nicht erreichbar:**
```bash
# Service Status
sudo systemctl status nobelbrett-dashboard

# Dashboard Logs
sudo journalctl -u nobelbrett-dashboard -f

# Port prüfen
sudo netstat -tlnp | grep :8000
```

### ⏱️ Zeitplan

**DNS Propagation:** 1-24 Stunden (meist 1-2 Stunden)
**SSL-Zertifikat:** Sofort nach DNS-Propagation
**Dashboard:** Sofort verfügbar

### 📞 Support Kontakte

**IONOS Support:** 
- Tel: 0721 / 254 111
- https://www.ionos.de/hilfe/

**Let's Encrypt:** 
- https://letsencrypt.org/docs/

---

✅ **Nach erfolgreichem Setup ist das Dashboard unter https://www.nobelbrett.de erreichbar!** 