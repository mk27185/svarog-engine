import os
import requests
from dotenv import load_dotenv

load_dotenv()

class OpenTopographyClient:
    def __init__(self):
        self.api_key = os.getenv("OPENTOPOGRAPHY_API_KEY")
        if not self.api_key:
            raise ValueError("OpenTopography API key not found in .env")
        self.base_url = "https://portal.opentopography.org"

    def get_dem(self, bbox, output_format="GTiff") -> str:
        """
        Fetches DEM and saves it to a file.
        Returns the path to the saved .tif file.
        """
        min_lat, min_lon, max_lat, max_lon = bbox
        
        # Create a filename based on the bbox
        filename = f"dem_{min_lat:.4f}_{min_lon:.4f}_{max_lat:.4f}_{max_lon:.4f}.tif"
        file_path = os.path.join("data/downloads", filename)
        os.makedirs("data/downloads", exist_ok=True)

        endpoint = f"{self.base_url}/API/globaldem"
        params = {
            "demtype": "SRTMGL1",
            "south": min_lat,
            "north": max_lat,
            "west": min_lon,
            "east": max_lon,
            "outputFormat": output_format,
            "API_Key": self.api_key
        }

        try:
            response = requests.get(endpoint, params=params, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return file_path
        except requests.exceptions.RequestException as e:
            print(f"Error fetching DEM: {e}")
            return None


if __name__ == "__main__":
    # Test implementation
    client = OpenTopographyClient()
    # Example Bounding Box (small area)
    test_bbox = (49.9, 14.9, 50.0, 15.0)
    print(f"Testing DEM fetch for bbox: {test_bbox}")
    data = client.get_dem(test_bbox)
    if data:
        size_bytes = os.path.getsize(data)
        print(f"Successfully fetched DEM data, saved to: {data} (size: {size_bytes} bytes)")
    else:
        print("Failed to fetch DEM data.")
