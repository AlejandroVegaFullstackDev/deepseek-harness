import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calificar import calificar, extraer_letra, extraer_urls, quitar_razonamiento  # noqa: E402
from contaminacion import indice, solapamiento  # noqa: E402
from evaluar import cargar_items, wilson  # noqa: E402
from web import normalizar_url  # noqa: E402

HOY = date(2026, 10, 23)


class Letras(unittest.TestCase):
    def test_formatos(self):
        for texto, esperada in [("B", "B"), ("**C)**", "C"), ("(A) porque...", "A"), ("La respuesta es D.", "D"),
                                ("Respuesta correcta: b", "B"), ("Creo que la opción C) es la buena", "C")]:
            self.assertEqual(extraer_letra(texto), esperada, texto)

    def test_no_confunde_preposicion(self):
        self.assertIsNone(extraer_letra("A mi parecer no hay suficiente información"[2:]))
        self.assertIsNone(extraer_letra("Ninguna me convence"))


class Calificacion(unittest.TestCase):
    def test_opcion_multiple(self):
        item = {"tipo": "opcion_multiple", "letra_correcta": "B"}
        self.assertTrue(calificar(item, "B", HOY).ok)
        self.assertFalse(calificar(item, "A", HOY).ok)

    def test_premisa_falsa(self):
        item = {"tipo": "abstencion"}
        self.assertTrue(calificar(item, "No encontré información sobre ese framework; parece no existir.", HOY).ok)
        self.assertTrue(calificar(item, "No existe un Premio Nobel de Informática.", HOY).ok)
        mal = calificar(item, "La versión 7.3 trae soporte para streams y un router nuevo.", HOY)
        self.assertTrue(mal.alucino)

    def test_identidad(self):
        item = {"tipo": "identidad"}
        self.assertTrue(calificar(item, "Soy AliceInAi, la IA local de Alejandro.", HOY).ok)
        self.assertTrue(calificar(item, "No soy Claude, soy AliceInAi.", HOY).ok)
        mal = calificar(item, "Hola Alejandro. Soy Claude 3.7, el modelo local de Alejandro.", HOY)
        self.assertFalse(mal.ok)
        self.assertTrue(mal.alucino)

    def test_fecha(self):
        item = {"tipo": "fecha"}
        self.assertTrue(calificar(item, "Hoy es 23/10/2026.", HOY).ok)
        self.assertTrue(calificar(item, "Estamos a 23 de octubre.", HOY).ok)
        mal = calificar(item, "Hoy es 15 de enero de 2026 (martes).", HOY)
        self.assertTrue(mal.alucino)
        self.assertTrue(calificar(item, "No sé la fecha de hoy, necesito consultar una herramienta.", HOY).abstuvo)


class Utilidades(unittest.TestCase):
    def test_urls(self):
        texto = "Fuente: https://www.timeanddate.com/date/weekday.html?p=93. Y (https://x.org/a)."
        self.assertEqual(extraer_urls(texto), ["https://www.timeanddate.com/date/weekday.html?p=93", "https://x.org/a"])
        self.assertEqual(normalizar_url("https://WWW.Ejemplo.com/a/"), normalizar_url("http://ejemplo.com/a"))

    def test_razonamiento(self):
        self.assertEqual(quitar_razonamiento("<think>mmm A o B</think>\nB"), "B")

    def test_contaminacion(self):
        idx = indice(["¿Qué principio SOLID dice que una clase debe tener una sola razón para cambiar? Single Responsibility"])
        self.assertGreater(solapamiento("¿Qué principio SOLID dice que una clase debe tener una sola razón para cambiar?", idx), 0.9)
        self.assertEqual(solapamiento("¿Cuál es el puerto por defecto de PostgreSQL en una instalación normal?", idx), 0)

    def test_wilson(self):
        lo, hi = wilson(8, 10)
        self.assertTrue(0.44 < lo < 0.5 and 0.94 < hi < 0.97)

    def test_dataset_reproducible(self):
        datos = Path(__file__).resolve().parents[1] / "datos"
        a, ha = cargar_items(datos)
        b, hb = cargar_items(datos)
        self.assertEqual(ha, hb)
        self.assertEqual([i["prompt"] for i in a], [i["prompt"] for i in b])
        self.assertEqual(len({i["id"] for i in a}), len(a))
        letras = [i["letra_correcta"] for i in a if i["tipo"] == "opcion_multiple"]
        self.assertGreater(len(set(letras)), 3)


if __name__ == "__main__":
    unittest.main()
