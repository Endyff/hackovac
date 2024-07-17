# Háčkovač

Algoritmus generující předlohy pro filetové háčkování.

# Install

```
pip install -r requirements.txt
```

# Run

```
streamlit run hackovac.py --server.address=0.0.0.0 --server.port=4000 --browser.serverAddress 127.0.0.1 --server.runOnSave true
```

# Docker

```
docker build -t hackovac .

docker run --rm -p 4000:4000 hackovac

docker run -d --name hackovac -p 4000:4000 --restart=unless-stopped hackovac
docker exec -ti hackovac /bin/bash
```
