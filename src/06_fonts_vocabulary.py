# ===== CELL 15 (9.2): Fonts, road vocabulary, label rules =====
ROAD_DOWNLOAD_FONTS = True  # ~30 MB of OFL/Apache fonts from github.com/google/fonts: far more sign-like variety
ROAD_FONT_DIR = ROAD_DOWNLOAD_DIR / "fonts"
GOOGLE_FONT_FILES = [
    "apache/robotoslab/RobotoSlab[wght].ttf", "ofl/alfaslabone/AlfaSlabOne-Regular.ttf", "ofl/amita/Amita-Bold.ttf",
    "ofl/amita/Amita-Regular.ttf", "ofl/anekdevanagari/AnekDevanagari[wdth,wght].ttf", "ofl/anton/Anton-Regular.ttf",
    "ofl/archivonarrow/ArchivoNarrow[wght].ttf", "ofl/arya/Arya-Bold.ttf", "ofl/arya/Arya-Regular.ttf",
    "ofl/asar/Asar-Regular.ttf", "ofl/baloo2/Baloo2[wght].ttf", "ofl/barlow/Barlow-Bold.ttf",
    "ofl/barlowcondensed/BarlowCondensed-Bold.ttf", "ofl/bebasneue/BebasNeue-Regular.ttf", "ofl/biryani/Biryani-Bold.ttf",
    "ofl/biryani/Biryani-Regular.ttf", "ofl/blackopsone/BlackOpsOne-Regular.ttf", "ofl/cambay/Cambay-Bold.ttf",
    "ofl/cambay/Cambay-Regular.ttf", "ofl/dekko/Dekko-Regular.ttf", "ofl/eczar/Eczar[wght].ttf",
    "ofl/firasans/FiraSans-Bold.ttf", "ofl/glegoo/Glegoo-Regular.ttf", "ofl/gotu/Gotu-Regular.ttf",
    "ofl/halant/Halant-Bold.ttf", "ofl/halant/Halant-Regular.ttf", "ofl/hind/Hind-Bold.ttf",
    "ofl/hind/Hind-Regular.ttf", "ofl/hind/Hind-SemiBold.ttf", "ofl/ibmplexsansdevanagari/IBMPlexSansDevanagari-Bold.ttf",
    "ofl/ibmplexsansdevanagari/IBMPlexSansDevanagari-Regular.ttf", "ofl/inknutantiqua/InknutAntiqua-Regular.ttf", "ofl/jaldi/Jaldi-Bold.ttf",
    "ofl/jaldi/Jaldi-Regular.ttf", "ofl/kadwa/Kadwa-Bold.ttf", "ofl/kadwa/Kadwa-Regular.ttf",
    "ofl/kalam/Kalam-Bold.ttf", "ofl/kalam/Kalam-Regular.ttf", "ofl/karma/Karma-Bold.ttf",
    "ofl/karma/Karma-Regular.ttf", "ofl/khand/Khand-Bold.ttf", "ofl/khand/Khand-Regular.ttf",
    "ofl/khula/Khula-Bold.ttf", "ofl/khula/Khula-Regular.ttf", "ofl/kurale/Kurale-Regular.ttf",
    "ofl/laila/Laila-Bold.ttf", "ofl/laila/Laila-Regular.ttf", "ofl/lato/Lato-Bold.ttf",
    "ofl/marcellus/Marcellus-Regular.ttf", "ofl/martel/Martel-Bold.ttf", "ofl/martel/Martel-Regular.ttf",
    "ofl/modak/Modak-Regular.ttf", "ofl/montserrat/Montserrat[wght].ttf", "ofl/mukta/Mukta-Bold.ttf",
    "ofl/mukta/Mukta-ExtraBold.ttf", "ofl/mukta/Mukta-Regular.ttf", "ofl/notosansdevanagari/NotoSansDevanagari[wdth,wght].ttf",
    "ofl/notoserifdevanagari/NotoSerifDevanagari[wdth,wght].ttf", "ofl/opensans/OpenSans[wdth,wght].ttf", "ofl/oswald/Oswald[wght].ttf",
    "ofl/overpass/Overpass[wght].ttf", "ofl/palanquin/Palanquin-Bold.ttf", "ofl/palanquin/Palanquin-Regular.ttf",
    "ofl/poppins/Poppins-Black.ttf", "ofl/poppins/Poppins-Bold.ttf", "ofl/poppins/Poppins-Regular.ttf",
    "ofl/pragatinarrow/PragatiNarrow-Bold.ttf", "ofl/pragatinarrow/PragatiNarrow-Regular.ttf", "ofl/ptsans/PT_Sans-Web-Bold.ttf",
    "ofl/rajdhani/Rajdhani-Bold.ttf", "ofl/rajdhani/Rajdhani-Regular.ttf", "ofl/rhodiumlibre/RhodiumLibre-Regular.ttf",
    "ofl/robotocondensed/RobotoCondensed[wght].ttf", "ofl/rozhaone/RozhaOne-Regular.ttf", "ofl/sahitya/Sahitya-Regular.ttf",
    "ofl/sarala/Sarala-Bold.ttf", "ofl/sarala/Sarala-Regular.ttf", "ofl/sarpanch/Sarpanch-Bold.ttf",
    "ofl/sarpanch/Sarpanch-Regular.ttf", "ofl/sourcesans3/SourceSans3[wght].ttf", "ofl/sumana/Sumana-Regular.ttf",
    "ofl/sura/Sura-Bold.ttf", "ofl/sura/Sura-Regular.ttf", "ofl/teko/Teko[wght].ttf",
    "ofl/tillana/Tillana-Bold.ttf", "ofl/tillana/Tillana-Regular.ttf", "ofl/tirodevanagarihindi/TiroDevanagariHindi-Regular.ttf",
    "ofl/vesperlibre/VesperLibre-Regular.ttf", "ofl/yatraone/YatraOne-Regular.ttf", "ufl/ubuntu/Ubuntu-Bold.ttf",
]


def download_google_fonts(dest, files=GOOGLE_FONT_FILES):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    ok = 0
    for rel in files:
        out = dest / Path(rel).name
        if out.is_file() and out.stat().st_size > 1000:
            ok += 1
            continue
        url = "https://raw.githubusercontent.com/google/fonts/main/" + urllib.parse.quote(rel)
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            if len(data) < 1000:
                raise ValueError("unexpectedly small file")
            part = out.with_name(out.name + ".part")
            part.write_bytes(data)
            part.replace(out)
            ok += 1
        except Exception as exc:  # A missing optional font must not stop the notebook.
            print("Skipped font", rel, "-", exc)
    return ok


_ODD_FONT_WORDS = ("unifont", "ipag", "ipam", "ipaex", "loma", "kinnari", "garuda", "norasi", "purisa",
                   "sawasdee", "umpush", "waree", "tlwg", "mono", "emoji", "symbol", "math", "music",
                   "cjk", "dingbat", "fallback", "lklug", "humor")


def build_road_fonts(font_dir=None):
    """Latin/Devanagari font pools. Uses the notebook's find_fonts (it also checks Pillow RAQM)."""
    fonts = find_fonts(str(font_dir) if font_dir else None, hindi=True)

    def normal(path):
        name = Path(path).name.lower()
        return not any(word in name for word in _ODD_FONT_WORDS)

    latin = [p for p in fonts["latin"] if normal(p)] or fonts["latin"]
    deva = [p for p in fonts["deva"] if normal(p)] or fonts["deva"]
    return {"latin": latin, "deva": deva}


if ROAD_DOWNLOAD_FONTS:
    print("Fonts available from google/fonts:", download_google_fonts(ROAD_FONT_DIR), "of", len(GOOGLE_FONT_FILES))
ROAD_FONTS = build_road_fonts(ROAD_FONT_DIR if ROAD_FONT_DIR.is_dir() else None)
print(f"Road fonts: {len(ROAD_FONTS['latin'])} Latin, {len(ROAD_FONTS['deva'])} Devanagari")

# ---------------------------------------------------------------- vocabulary (plain word lists, no model)
ROAD_EN_PLACES = """AGRA ALIGARH ALLAHABAD AMETHI AMRITSAR AYODHYA AZAMGARH BAHRAICH BALLIA BANDA BAREILLY BASTI
BIJNOR BUDAUN BULANDSHAHR CHANDAULI CHITRAKOOT DEORIA ETAH ETAWAH FAIZABAD FARRUKHABAD FATEHPUR FIROZABAD
GHAZIABAD GHAZIPUR GONDA GORAKHPUR HAMIRPUR HAPUR HARDOI HATHRAS JALAUN JAUNPUR JHANSI KANNAUJ KANPUR
KAUSHAMBI KUSHINAGAR LAKHIMPUR LALITPUR LUCKNOW MAHOBA MAINPURI MATHURA MAU MEERUT MIRZAPUR MORADABAD
MUZAFFARNAGAR NOIDA PILIBHIT PRATAPGARH PRAYAGRAJ RAEBARELI RAMPUR SAHARANPUR SHAHJAHANPUR SHAMLI SITAPUR
SONBHADRA SULTANPUR UNNAO VARANASI DELHI JAIPUR JODHPUR UDAIPUR KOTA AJMER BIKANER PATNA GAYA MUZAFFARPUR
BHAGALPUR RANCHI JAMSHEDPUR DHANBAD BHOPAL INDORE GWALIOR JABALPUR UJJAIN RAIPUR BILASPUR NAGPUR MUMBAI
PUNE NASHIK AURANGABAD KOLKATA HOWRAH ASANSOL CHENNAI MADURAI COIMBATORE HYDERABAD WARANGAL BENGALURU
MYSURU MANGALURU AHMEDABAD SURAT VADODARA RAJKOT CHANDIGARH LUDHIANA JALANDHAR PATIALA AMBALA PANIPAT
KARNAL ROHTAK HISAR GURUGRAM FARIDABAD DEHRADUN HARIDWAR RISHIKESH ROORKEE HALDWANI NAINITAL SHIMLA MANALI
JAMMU SRINAGAR GUWAHATI SHILLONG BHUBANESWAR CUTTACK PURI KOCHI THIRUVANANTHAPURAM GOA PANAJI""".split()
ROAD_EN_WORDS = """STOP GO SLOW SPEED LIMIT SCHOOL AHEAD HOSPITAL NO PARKING ENTRY EXIT ONE WAY KEEP LEFT RIGHT
TOLL PLAZA NATIONAL STATE HIGHWAY EXPRESSWAY BYPASS FLYOVER RAILWAY STATION BUS STAND AIRPORT POLICE PETROL
PUMP FUEL CNG DIVERSION WORK IN PROGRESS DRIVE SLOWLY ACCIDENT PRONE AREA SPEED BREAKER U TURN NARROW BRIDGE
HORN OK PLEASE WELCOME THANK YOU VISIT AGAIN GOVERNMENT OFFICE COLLEGE UNIVERSITY BANK ATM MEDICAL STORE
CLINIC PHARMACY CHEMIST HOTEL RESTAURANT DHABA SWEETS GENERAL KIRANA MOBILE ELECTRONICS HARDWARE CLOTH
HOUSE TAILORS BEAUTY PARLOUR SALON GYM COACHING ACADEMY PUBLIC INTER NAGAR PALIKA MUNICIPAL CORPORATION
DISTRICT TEHSIL BLOCK VILLAGE MARG ROAD CHOWK CROSSING GALI SADAR BAZAR MANDI COLONY VIHAR PURAM GANJ
MARKET CITY CENTRE JUNCTION CIRCLE GATE TEMPLE MOSQUE CHURCH PARK STADIUM MUSEUM FORT LAKE RIVER DAM CANAL
ZONE CAUTION DANGER WARNING EMERGENCY SAFETY FIRST HELMET SEAT BELT DRUNKEN DRIVING OVERTAKING PROHIBITED
PEDESTRIAN CYCLE TRACK LANE ONLY TRUCKS HEAVY VEHICLES MAXIMUM MINIMUM KM KMPH TONNES METRE FOOD PLAZA
TOILET REST AREA PARKING FREE PAID TICKET COUNTER ENQUIRY PLATFORM WAITING HALL CARGO TERMINAL DEPARTURE
ARRIVAL TAXI AUTO STAND METRO DEPOT WORKSHOP SERVICE CENTRE TYRES BATTERY GARAGE SPARE PARTS AGENCY
TRADERS ENTERPRISES INDUSTRIES CEMENT STEEL PAINTS PLYWOOD FURNITURE JEWELLERS OPTICALS DENTAL CARE
NURSING HOME DIAGNOSTIC LAB PATHOLOGY BLOOD BANK AMBULANCE FIRE POST INDIA BHARAT NEW OLD MAIN NORTH SOUTH
EAST WEST UPPER LOWER DAILY OPEN CLOSED SALE OFFER DISCOUNT BEST QUALITY PURE VEG NON FAMILY SHOP MART
STUDIO PHOTO COPY PRINT CYBER CAFE INTERNET RECHARGE WATER MILK DAIRY BAKERY FRUITS VEGETABLES""".split()
ROAD_EN_BRANDS = """SBI HDFC ICICI PNB AXIS BOB CANARA AIRTEL JIO VI BSNL TATA AMUL MOTHER DAIRY HERO HONDA BAJAJ TVS
MARUTI SUZUKI MAHINDRA HYUNDAI INDIAN OIL BHARAT PETROLEUM HP NAYARA APOLLO FORTIS LIC IRCTC NHAI UPSRTC""".split()
ROAD_HI_PLACES = """आगरा अलीगढ़ प्रयागराज अमेठी अमृतसर अयोध्या आज़मगढ़ बहराइच बलिया बांदा बरेली बस्ती बिजनौर बदायूँ
बुलंदशहर चंदौली चित्रकूट देवरिया एटा इटावा फर्रुखाबाद फतेहपुर फिरोज़ाबाद गाज़ियाबाद गाज़ीपुर गोंडा गोरखपुर
हमीरपुर हापुड़ हरदोई हाथरस जालौन जौनपुर झाँसी कन्नौज कानपुर कौशाम्बी कुशीनगर लखीमपुर ललितपुर लखनऊ महोबा
मैनपुरी मथुरा मऊ मेरठ मिर्ज़ापुर मुरादाबाद मुज़फ्फरनगर नोएडा पीलीभीत प्रतापगढ़ रायबरेली रामपुर सहारनपुर शाहजहाँपुर
शामली सीतापुर सोनभद्र सुल्तानपुर उन्नाव वाराणसी दिल्ली जयपुर जोधपुर उदयपुर कोटा अजमेर बीकानेर पटना गया
मुज़फ्फरपुर भागलपुर राँची जमशेदपुर धनबाद भोपाल इंदौर ग्वालियर जबलपुर उज्जैन रायपुर बिलासपुर नागपुर मुंबई पुणे
नासिक कोलकाता चेन्नई हैदराबाद बेंगलुरु अहमदाबाद सूरत वडोदरा चंडीगढ़ लुधियाना जालंधर पटियाला अंबाला पानीपत करनाल
रोहतक हिसार गुरुग्राम फरीदाबाद देहरादून हरिद्वार ऋषिकेश रुड़की हल्द्वानी नैनीताल शिमला मनाली जम्मू श्रीनगर गुवाहाटी
भुवनेश्वर पुरी""".split()
ROAD_HI_WORDS = """रुकें धीरे चलें गति सीमा विद्यालय आगे अस्पताल पार्किंग निषेध प्रवेश निकास एकतरफा मार्ग बाएँ दाएँ
टोल प्लाज़ा राष्ट्रीय राजमार्ग राज्य एक्सप्रेसवे बाईपास फ्लाईओवर रेलवे स्टेशन बस स्टैंड हवाई अड्डा पुलिस थाना
पेट्रोल पंप किमी कि.मी. परिवर्तन कार्य प्रगति पर है दुर्घटना संभावित क्षेत्र स्पीड ब्रेकर स्वागत धन्यवाद पुनः
पधारें सरकारी कार्यालय महाविद्यालय विश्वविद्यालय बैंक मेडिकल स्टोर क्लीनिक होटल रेस्टोरेंट ढाबा मिष्ठान भंडार
जनरल किराना मोबाइल इलेक्ट्रॉनिक्स हार्डवेयर वस्त्रालय टेलर्स ब्यूटी पार्लर सैलून जिम कोचिंग एकेडमी पब्लिक स्कूल
इंटर कॉलेज नगर पालिका निगम जनपद तहसील विकास खंड ग्राम रोड चौक चौराहा गली सदर बाज़ार मंडी कॉलोनी विहार
पुरम गंज जिला उत्तर प्रदेश भारत सरकार शिक्षा स्वास्थ्य केंद्र प्राथमिक माध्यमिक आंगनबाड़ी पंचायत भवन डाकघर
बिजली विभाग जल पुलिस चौकी सावधान खतरा मोड़ पुल संकरा मंदिर मस्जिद गुरुद्वारा पार्क स्टेडियम संग्रहालय किला
झील नदी नहर बाँध क्षेत्र आपातकाल सुरक्षा पहले हेलमेट सीट बेल्ट शराब पीकर वाहन न चलाएँ ओवरटेक करना मना है
पैदल यात्री साइकिल केवल ट्रक भारी वाहन अधिकतम न्यूनतम टन मीटर भोजनालय शौचालय विश्राम स्थल निःशुल्क टिकट
घर पूछताछ प्लेटफार्म प्रतीक्षालय प्रस्थान आगमन टैक्सी ऑटो मेट्रो डिपो सर्विस सेंटर टायर बैटरी गैराज एजेंसी
ट्रेडर्स इंटरप्राइजेज उद्योग सीमेंट स्टील पेंट्स फर्नीचर ज्वैलर्स ऑप्टिकल्स दंत चिकित्सा नर्सिंग होम जाँच पैथोलॉजी
रक्त कोष एम्बुलेंस अग्निशमन नया पुराना मुख्य उत्तर दक्षिण पूर्व पश्चिम खुला बंद सेल छूट शुद्ध शाकाहारी परिवार
दुकान स्टूडियो फोटो कॉपी साइबर कैफ़े रिचार्ज पानी दूध डेयरी बेकरी फल सब्ज़ी श्री श्रीमती कृपया ध्यान दें
कृषि मंडल सहकारी समिति उचित मूल्य राशन दवाखाना पशु चिकित्सालय प्रखंड न्यायालय कचहरी जिलाधिकारी""".split()
DEVA_DIGITS = "०१२३४५६७८९"
_DV_CONS = "कखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह"
_DV_NUKTA_OK = "कखगजडढफ"
_DV_VOWELS = "अआइईउऊएऐओऔऋ"
_DV_SIGNS = ["", "", "", "ा", "ि", "ी", "ु", "ू", "े", "ै", "ो", "ौ", "ृ"]


def random_deva_word(rng, syllables=None):
    """Pseudo-words from random valid Devanagari syllables (teaches unseen conjunct/matra shapes)."""
    n = int(syllables or rng.integers(1, 5))
    out = []
    for i in range(n):
        if i == 0 and rng.random() < 0.15:
            out.append(str(rng.choice(list(_DV_VOWELS))) + ("ं" if rng.random() < 0.15 else ""))
            continue
        cluster = str(rng.choice(list(_DV_CONS)))
        if cluster in _DV_NUKTA_OK and rng.random() < 0.08:
            cluster += "़"
        if rng.random() < 0.18:
            cluster += "्" + str(rng.choice(list(_DV_CONS)))
        if rng.random() < 0.05:
            cluster = "र्" + cluster
        syllable = cluster + str(rng.choice(_DV_SIGNS))
        if rng.random() < 0.12:
            syllable += str(rng.choice(["ं", "ँ"]))
        out.append(syllable)
    if rng.random() < 0.05:
        out.append("ः")
    return "".join(out)


def _to_deva_digits(text):
    return "".join(DEVA_DIGITS[int(c)] if c.isdigit() else c for c in text)


def random_number_text(rng, hindi=False):
    kind = rng.random()
    if kind < 0.25:
        text = str(int(rng.integers(1, 999)))
    elif kind < 0.45:
        text = f"{int(rng.integers(6, 10))}{int(rng.integers(0, 10**9)):09d}"  # phone number
    elif kind < 0.6:
        text = f"{rng.choice(['NH', 'SH', 'MDR'])}{rng.choice(['-', ' ', ''])}{int(rng.integers(1, 999))}"
    elif kind < 0.72:
        text = f"0{int(rng.integers(100, 999))}-{int(rng.integers(100000, 9999999))}"
    elif kind < 0.84:
        text = f"{int(rng.integers(1, 300))}{rng.choice(['km', 'KM', 'Km', ' km'])}"
    elif kind < 0.92:
        text = f"{rng.choice(['UP', 'DL', 'HR', 'MP', 'RJ', 'BR'])}{int(rng.integers(1, 99)):02d}" \
               f"{''.join(rng.choice(list(string.ascii_uppercase), 2))}{int(rng.integers(1, 9999)):04d}"
    else:
        text = f"{int(rng.integers(1950, 2027))}"
    if hindi and rng.random() < 0.35:
        text = _to_deva_digits(text)
    return text


def random_latin_string(rng, n=None):
    n = int(n or rng.integers(2, 11))
    chars = string.ascii_uppercase if rng.random() < 0.6 else string.ascii_letters + string.digits
    return "".join(rng.choice(list(chars), n))


def road_word(rng, lang):
    """One road-side word. lang: 'hindi' | 'english' | 'digits'."""
    r = rng.random()
    if lang == "hindi":
        if r < 0.40:
            return str(rng.choice(ROAD_HI_WORDS))
        if r < 0.70:
            return str(rng.choice(ROAD_HI_PLACES))
        if r < 0.80:
            return str(rng.choice(HINDI_WORDS))
        return random_deva_word(rng)
    if lang == "digits":
        return random_number_text(rng, hindi=bool(rng.random() < 0.3))
    if r < 0.40:
        word = str(rng.choice(ROAD_EN_WORDS))
    elif r < 0.68:
        word = str(rng.choice(ROAD_EN_PLACES))
    elif r < 0.78:
        word = str(rng.choice(ROAD_EN_BRANDS))
    elif r < 0.86:
        word = str(rng.choice(ENGLISH_WORDS))
    else:
        return random_latin_string(rng)
    style = rng.random()
    return word.title() if style < 0.3 else word.lower() if style < 0.4 else word


def extend_road_vocab(texts, hindi_list=None, english_list=None):
    """Add real training-split words (phase 2) so synthetic crops also cover real vocabulary."""
    added = 0
    for text in texts:
        for word in text.split():
            if road_clean_label(word) is None or len(word) > 24:
                continue
            target = ROAD_HI_WORDS if any("\u0900" <= c <= "\u097f" for c in word) else ROAD_EN_WORDS
            target.append(word)
            added += 1
    return added


# ---------------------------------------------------------------- label normalisation
_ROAD_CHAR_MAP = {"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-",
                  "\u2014": "-", "\u2212": "-", "\u00a0": " ", "\u00b4": "'", "\u2026": "...", "\u0060": "'"}
_ROAD_ZERO_WIDTH = set("\u200b\u200c\u200d\u2060\ufeff")
ROAD_CHARSET = set(CHARS)


def _road_basic_clean(text):
    text = unicodedata.normalize("NFC", str(text))
    text = "".join(_ROAD_CHAR_MAP.get(c, c) for c in text if c not in _ROAD_ZERO_WIDTH)
    return " ".join(text.split())


def road_clean_label(text):
    """Training label, or None if empty/'###'/contains characters the model cannot output."""
    if text is None:
        return None
    text = _road_basic_clean(text)
    if not text or set(text) <= {"#"} or any(c not in ROAD_CHARSET for c in text):
        return None
    return text


def road_match_key(text):
    """Comparison key for 'read correctly': case-insensitive, ignores spaces and punctuation."""
    text = _road_basic_clean(text or "").casefold()
    return "".join(c for c in text if c.isalnum() or unicodedata.category(c) in ("Mn", "Mc"))


_vocab_rng = np.random.default_rng(0)
_bad = [w for w in ROAD_HI_WORDS + ROAD_HI_PLACES + ROAD_EN_WORDS + ROAD_EN_PLACES
        + [random_deva_word(_vocab_rng) for _ in range(300)] if road_clean_label(w) is None]
if _bad:
    raise ValueError(f"Vocabulary contains characters outside CHARS: {_bad[:5]}")
print("Vocabulary:", len(ROAD_EN_WORDS) + len(ROAD_EN_PLACES), "English,",
      len(ROAD_HI_WORDS) + len(ROAD_HI_PLACES), "Hindi words (+ random pseudo-words and numbers)")
