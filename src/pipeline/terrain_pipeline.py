import os

class TerrainPipeline:
    """
    Orchestrátor pro celý proces: stahování -> konverze -> výsledek.
    """
    def __init__(self, client, converter):
        self.client = client
        self.converter = converter

    def run_pipeline(self, bbox, output_base_name):
        """
        Spustí celý pipeline pro daný bounding box.
        """
        print(f"--- Start Pipeline for {output_base_name} ---")
        
        try:
            # 1. Extrakce (Stahování DEM)
            print(f"1. Stahuji DEM pro bounding box: {bbox}...")
            tif_path = self.client.get_dem(bbox)
            
            if not tif_path or not os.path.exists(tif_path):
                print("Chyba: Nepodařilo se stáhnout DEM.")
                return None

            print(f"   Úspěšně staženo: {tif_path}")

            # 2. Konverze (TIF -> OBJ)
            print(f"2. Převádím na 3D mesh OBJ...")
            obj_path = self.converter.convert_tif_to_obj(tif_path, obj_name=output_base_name)
            
            print(f"   Úspěšná konverze: {obj_path}")
            print(f"--- Pipeline Finished Successfully ---")
            return obj_path

        except Exception as e:
            print(f"--- Pipeline FAILED ---")
            print(f"Chyba: {e}")
            return None

def main():
    import sys
    import os
    sys.path.append('.')
    
    try:
        from src.extraction.opentopography_client import OpenTopographyClient
        from src.conversion.terrain_converter import TerrainConverter
    except ImportError as e:
        print(f"Import Error: {e}")
        return

    client = OpenTopographyClient()
    converter = TerrainConverter(output_dir='outputs/test_results')
    pipeline = TerrainPipeline(client, converter)
    
    # Testovací oblast: Praha
    test_bbox = (50.07, 14.43, 50.08, 14.44)
    pipeline.run_pipeline(test_bbox, 'prague_test_area')

if __name__ == "__main__":
    main()
