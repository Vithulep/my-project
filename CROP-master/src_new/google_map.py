# import googlemaps
# from datetime import datetime

# gmaps = googlemaps.Client(key='AIzaSyDPV0Aze7-htGc4IiYPryFeIHncBzTNwqo')

# # Geocoding an address
# geocode_result = gmaps.geocode('1600 Amphitheatre Parkway, Mountain View, CA')

# # Look up an address with reverse geocoding
# reverse_geocode_result = gmaps.reverse_geocode((40.714224, -73.961452))

# # Request directions via public transit
# now = datetime.now()
# directions_result = gmaps.directions("Sydney Town Hall",
#                                      "Parramatta, NSW",
#                                      mode="transit",
#                                      departure_time=now)

# # Validate an address with address validation
# addressvalidation_result =  gmaps.addressvalidation(['1600 Amphitheatre Pk'], 
#                                                     regionCode='US',
#                                                     locality='Mountain View', 
#                                                     enableUspsCass=True)

# print(directions_result) 

# import googlemaps
# from datetime import datetime

# gmaps = googlemaps.Client(key='AIzaSyDPV0Aze7-htGc4IiYPryFeIHncBzTNwqo')

# # Geocoding an address
# geocode_result = gmaps.geocode('1600 Amphitheatre Parkway, Mountain View, CA')
# # print("Geocoded Location:", geocode_result[0]['formatted_address'])

# # Look up an address with reverse geocoding
# reverse_geocode_result = gmaps.reverse_geocode((40.714224, -73.961452))
# # print("Reverse Geocoded Location:", reverse_geocode_result[0]['formatted_address'])

# # Request directions via public transit
# now = datetime.now()
# directions_result = gmaps.directions("Bengaluru, Karnataka",
#                                      "Bagalkote, Karnataka",
#                                      mode="driving",
#                                      departure_time=now)
# print("Transit Directions:")
# for step in directions_result[0]['legs'][0]['steps']:
#     print(step['html_instructions'])
#     print("Distance:", step['distance']['text'])

# # Validate an address with address validation
# addressvalidation_result = gmaps.addressvalidation(['1600 Amphitheatre Pk'],
#                                                     regionCode='US',
#                                                     locality='Mountain View',
#                                                     enableUspsCass=True)
# # print("Validated Address:", addressvalidation_result['candidates'][0]['formatted_address'])

# total_distance = 0.0
# for step in directions_result[0]['legs'][0]['steps']:
#     if 'transit_details' in step and step['transit_details']['line']['vehicle']['type'] in ['BUS', 'TRUCK']:
#         total_distance += step['distance']['value']

# print("Total Distance for Bus/Truck:", total_distance / 1000, "km")

import googlemaps

def calculate_road_distance(api_key, origin, destination):
    gmaps = googlemaps.Client(key=api_key)

    # Request distance matrix
    distance_matrix = gmaps.distance_matrix(origin, destination, mode="driving")

    # Check if the response is valid
    if distance_matrix['status'] == 'OK':
        # Extract the distance in meters
        distance_in_meters = distance_matrix['rows'][0]['elements'][0]['distance']['value']
        
        # Convert meters to kilometers
        distance_in_kilometers = distance_in_meters / 1000

        print(f"Road Distance between {origin} and {destination}: {distance_in_kilometers:.2f} km")
    else:
        print(f"Error: {distance_matrix['status']}")

# Replace 'Your-API-Key' with your actual Google Maps API key
api_key = 'AIzaSyDPV0Aze7-htGc4IiYPryFeIHncBzTNwqo'
origin = 'Bengaluru, Karnataka'
destination = 'Bagalkote, Karnataka'



calculate_road_distance(api_key, origin, destination)

