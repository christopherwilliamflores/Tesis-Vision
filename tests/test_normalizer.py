import unicodedata

from app.services.normalizer import ProductTextNormalizer


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def test_normalizer_ignores_octagon_warning_as_product_type() -> None:
    text = "Nestle\nMorochas\nSNACK\nALTO EN AZUCAR\nEVITAR SU CONSUMO EXCESIVO"

    result = ProductTextNormalizer().normalize(text, source_name="morochas.png")

    assert _fold(result.tipo_producto or "") != "azucar"
    assert result.categoria_sugerida == "snacks y golosinas"


def test_normalizer_ignores_split_octagon_warning_fragments() -> None:
    text = "Morochas\nALTO EN\nGRASAS SATURADAS\nALTO EN\nAZUCAR\nSNACK"

    result = ProductTextNormalizer().normalize(text, source_name="morochas.png")

    assert _fold(result.tipo_producto or "") != "azucar"
    assert result.categoria_sugerida == "snacks y golosinas"


def test_normalizer_ignores_joined_octagon_ocr_noise() -> None:
    text = (
        "ALTOEN\nALTOEN\nGRASAS\nSATURADAS\nAZUCAR\nLVITAHSUCONSUIO\nETCISVO\n"
        "Nestla\nMorochas\nSNACK\nSiua\nRE8\nperciaes\n429\n71\n16\n"
        "Oaetassahoravanilla\nbalndscoapataabo.chacolate"
    )

    result = ProductTextNormalizer().normalize(text, source_name="morochas.png")

    assert result.nombre_producto == "Morochas Snack Nestlé"
    assert result.marca == "Nestlé"
    assert _fold(result.tipo_producto or "") == "snack"
    assert result.contenido_neto == "42 g"
    assert result.unidad_medida == "g"
    assert result.categoria_sugerida == "snacks y golosinas"


def test_normalizer_detects_peruvian_brand_content_and_category() -> None:
    text = "GLORIA\nLECHE EVAPORADA ENTERA\nContenido Neto 400 g"

    result = ProductTextNormalizer().normalize(text)

    assert result.marca == "Gloria"
    assert result.tipo_producto == "Leche"
    assert result.contenido_neto == "400 g"
    assert result.unidad_medida == "g"
    assert result.categoria_sugerida == "lacteos"
    assert result.nombre_producto is not None


def test_normalizer_handles_common_beverage_units() -> None:
    text = "Inca Kola\nGaseosa sabor nacional\nBotella 1.5 L"

    result = ProductTextNormalizer().normalize(text)

    assert result.marca == "Inca Kola"
    assert result.tipo_producto == "Gaseosa"
    assert result.contenido_neto == "1.5 L"
    assert result.presentacion == "botella 1.5 L"
    assert result.categoria_sugerida == "bebidas"


def test_normalizer_prioritizes_fanta_over_packaging_noise() -> None:
    text = "CAFFEINE\n100%\nFREE\nfanta\nORANGE"

    result = ProductTextNormalizer().normalize(text, source_name="0020_fanta_na.png")

    assert result.nombre_producto == "Gaseosa Fanta Naranja"
    assert result.marca == "Fanta"
    assert result.tipo_producto == "Gaseosa"
    assert result.categoria_sugerida == "bebidas"


def test_normalizer_identifies_yogurt_as_product_type_not_only_brand() -> None:
    text = "GLORIA\nYogurt fresa\n1 kg"

    result = ProductTextNormalizer().normalize(text)

    assert result.nombre_producto == "Yogurt Gloria Fresa"
    assert result.marca == "Gloria"
    assert result.tipo_producto == "Yogurt"
    assert result.categoria_sugerida == "lacteos"


def test_normalizer_identifies_mantequilla_as_product_type() -> None:
    text = "Mantequilla Gloria con sal 200 g"

    result = ProductTextNormalizer().normalize(text)

    assert result.nombre_producto == "Mantequilla Gloria"
    assert result.marca == "Gloria"
    assert result.tipo_producto == "Mantequilla"


def test_normalizer_detects_bells_sugar_from_ocr_text() -> None:
    text = "Bell's\nAZUCAR\nBLANCA\nPESO NETO 5Kg"

    result = ProductTextNormalizer().normalize(text)

    assert result.nombre_producto == "Azúcar Bell's Blanca"
    assert result.marca == "Bell's"
    assert result.tipo_producto == "Azúcar"
    assert result.contenido_neto == "5 kg"
    assert result.unidad_medida == "kg"
    assert result.categoria_sugerida == "abarrotes"
    assert "No se identificó marca" not in " ".join(result.warnings)


def test_normalizer_detects_portugal_body_cream_from_joined_ocr_text() -> None:
    text = (
        "PORTUGAL\nCREMACORPORAL\ncoco fresh\nCONACEITEDECOCO\n"
        "Manteca de Karitey Vitamina E\nEXTRA HIDRATACION\n"
        "Ayudaameoraanvdau\ndolapel\nTODOTIPODEPIEL"
    )

    result = ProductTextNormalizer().normalize(text, source_name="portugal coco.png")

    assert result.nombre_producto == "Crema Corporal Portugal Coco Fresh"
    assert result.marca == "Portugal"
    assert result.tipo_producto == "Crema"
    assert result.categoria_sugerida == "cuidado personal"
    assert "No se identificó marca" not in " ".join(result.warnings)


def test_normalizer_repairs_nivea_body_cream_ocr_text() -> None:
    text = (
        "NIVEA\nCrema Corpora\nMilk\nNutritiva\ntr ctny.Hamectockn\n"
        "C85\nLprotundo\nDotoco\necaAmendronAMtomnoE"
    )

    result = ProductTextNormalizer().normalize(text, source_name="nivea.png")

    assert result.nombre_producto == "Crema Corporal Nivea Milk Nutritiva"
    assert result.marca == "Nivea"
    assert result.tipo_producto == "Crema"
    assert result.categoria_sugerida == "cuidado personal"
    assert "No se identificó tipo" not in " ".join(result.warnings)


def test_normalizer_detects_johnsons_soap_multipack() -> None:
    text = (
        "PACK x3JABONES\nbaby\njabón.cremoso\nconingredienteshidratantes\n"
        "pieldelicada\nlibre deparabenos\nyftalatos\n3 x 125g c/u\n"
        "hpoaleigenios\nPRCEADO\nPuedeperderhasta12.5gc/u"
    )

    result = ProductTextNormalizer().normalize(text, source_name="johnsons pack jabones.png")

    assert result.nombre_producto == "Jabón Cremoso Johnson's Bebé"
    assert result.marca == "Johnson's"
    assert result.tipo_producto == "Jabón"
    assert result.presentacion == "pack x 3"
    assert result.contenido_neto == "3 x 125 g"
    assert result.unidad_medida == "g"
    assert result.categoria_sugerida == "bebés y mamá"


def test_normalizer_detects_kolynos_loose_offer_pack() -> None:
    text = (
        "3\nCREMAS\nOFERTA\nESPECIAL\nDENTALES\n60mL\n"
        "Pvsprotoniedaalogrecipregufar sspendoCerniaal cubco\n"
        "CREHACENTALCOWFLCOR+CALCO\nKolynos\nSUPER\nBLANCO\n"
        "EMPAQUE\nFAMILIAR"
    )

    result = ProductTextNormalizer().normalize(text, source_name="kolinos.png")

    assert result.nombre_producto == "Pasta dental Kolynos Super Blanco"
    assert result.marca == "Kolynos"
    assert result.tipo_producto == "Pasta dental"
    assert result.presentacion == "pack x 3"
    assert result.contenido_neto == "3 x 60 ml"
    assert result.unidad_medida == "ml"
    assert result.categoria_sugerida == "cuidado personal"


def test_normalizer_detects_boreal_detergent_from_ocr_text() -> None:
    text = "boreal\nDetergente\nliquido\nAroma\nLawanda\nCo3L"

    result = ProductTextNormalizer().normalize(text, source_name="boreal.png")

    assert result.nombre_producto == "Detergente Boreal Lavanda"
    assert result.marca == "Boreal"
    assert result.tipo_producto == "Detergente"
    assert result.contenido_neto == "3 L"
    assert result.categoria_sugerida == "limpieza"


def test_normalizer_prefers_detected_bolivar_variant_over_filename_color() -> None:
    text = (
        "ABRIRAOU\nNUEVA\nIMAGEN\nDETERGENTE\nROSAS Y MAGNOLIAS\n"
        "Bolivar\nCuidadoy\nSuavidad\nSUAVIDAD Y\ncontogue\n"
        "FRAGANCIA\nde suav zante\nycapsulas\nPROLONGADA\ndearoma"
    )

    result = ProductTextNormalizer().normalize(
        text,
        source_name="bol-morado_png.rf.8deff10bb587d9bb8525ae403509c340.jpg",
    )

    assert result.nombre_producto == "Detergente Bolívar Rosas y Magnolias"
    assert result.marca == "Bolívar"
    assert result.tipo_producto == "Detergente"
    assert result.categoria_sugerida == "limpieza"


def test_normalizer_detects_bolivar_cuidado_total_variant() -> None:
    text = (
        "NUEVA\nIMAGEN\nDCJERSENTEF!CFJE\nw 4kg\nBolivar\n"
        "Cuidado\nTotal\nPROTEGE\ncontsarculas\nELCOLORY\n"
        "protictoras\nLASFIBRAS"
    )

    result = ProductTextNormalizer().normalize(text, source_name="bolivar.png")

    assert result.nombre_producto == "Jabón Bolívar Cuidado Total"
    assert result.marca == "Bolívar"
    assert result.tipo_producto == "Jabón"
    assert result.contenido_neto == "4 kg"
 
