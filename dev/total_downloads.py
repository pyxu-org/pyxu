import requests


def get_downloads_from_pepy(package_name):
    response = requests.get(f"https://api.pepy.tech/api/v2/projects/{package_name}")
    response.raise_for_status()
    return response.json()["total_downloads"]


def create_combined_badge_url(count):
    base_url = "https://img.shields.io/badge/total%20downloads-"
    color = "green"
    return f"{base_url}{count}-{color}.svg"


def main():
    pycsou_downloads = get_downloads_from_pepy("pycsou")
    pyxu_downloads = get_downloads_from_pepy("pyxu")
    combined_downloads = pycsou_downloads + pyxu_downloads

    badge_url = create_combined_badge_url(combined_downloads)
    print(f"Badge URL: {badge_url}")


if __name__ == "__main__":
    main()
