# Analiza mreže hrvatskog turizma

Projekt analize kompleksnih mreža izgrađen na stvarnim podacima
o hrvatskom turizmu koje objavljuje Državni zavod za statistiku (DZS).
Projekt modelira hrvatske gradove/općine kao **mrežu sličnosti
destinacija** i na nju primjenjuje cijeli alat kolegija Analiza
kompleksnih mreža.

## Dataset

Ovaj projekt koristi **službenu DZS statistiku turizma**:
stvarne, državno objavljene, slobodno dostupne i citirane podatke, u
punoj rezoluciji po gradu/općini.

## Izvor podataka

**Tablica 1.6** — *Kapaciteti smještaja, dolasci i noćenja turista,
Republika Hrvatska, NUTS 2013 – 2. razina, županije, gradovi, općine,
po mjesecima* (`BS_TU16.px`, baza `Turizam`, DZS PxWeb).

https://web.dzs.hr/PXWeb/Selection.aspx?px_path=Turizam__Dolasci+i+no%C4%87enja+turista+u+komercijalnim+smje%C5%A1tajnim+objektima&px_tableid=BS_TU16.px&px_language=en&px_db=Turizam

Ovo su podaci na razini **općine/grada** (ne samo županije) — mjesečni
dolasci turista, noćenja i kapacitet ležajeva za svaki hrvatski grad/
općine

## Konstrukcija mreže

- **Čvorovi**: hrvatski gradovi/općine. Atributi: županija, oznaka obalno/kontinentalno, ukupni
  dolasci/noćenja/ležajevi, dominantno emitivno tržište (gdje je dostupno).
  
- **Bridovi**: ne postoje izravni podaci o toku turista između općina na
  ovoj razini, pa bridovi povezuju općine sa sličnim **sezonskim profilom
  turizma** — 12-mjesečna distribucija noćenja svake općine (normalizirana
  da zbroj iznosi 1) je njen "otisak sezonalnosti" (seasonality fingerprint);
  težina brida je kosinusna sličnost između dva otiska, proriješena (sparsified)
  metodom top-k najbližih susjeda po čvoru. Ovo je standardna konstrukcija
  "mreže sličnosti destinacija" u istraživanju turističkih mreža, i koristi
  podatke u punoj rezoluciji po općini umjesto da se posegne za grubljim
  grafom.

## Reprodukcija

```bash
pip install -r requirements.txt

# 1. Preuzmi sirove DZS podatke

# 2. Izgradi obrađene tablice (atributi čvorova, otisci sezonalnosti)
python -m src.preprocessing

# 3. Izgradi mrežu sličnosti (objedinjeni + godišnji vremenski grafovi)
python -m src.network_construction

# 4. Pokreni testove
pytest tests/ -q

# 5. Prođi kroz bilježnice 00 do 05 redom