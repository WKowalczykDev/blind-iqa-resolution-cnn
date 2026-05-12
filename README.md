# Setup

## nbstripout — usuwanie outputów z notebooków przed commitem

Żeby outputy z `.ipynb` nie trafiały do gita, zainstaluj nbstripout lokalnie:

```bash
pipx install nbstripout
nbstripout --install
nbstripout --install --attributes .gitattributes
```

> Wymaga `pipx`. Jeśli nie masz: `sudo apt install pipx`

Uruchom **raz po sklonowaniu repo**. Outputy w plikach na dysku zostają nieruszone — do gita idzie tylko kod komórek.