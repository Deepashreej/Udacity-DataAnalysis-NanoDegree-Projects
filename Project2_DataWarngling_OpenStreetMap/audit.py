"""
Your task in this exercise has two steps:

- audit the OSMFILE and change the variable 'mapping' to reflect the changes needed to fix 
    the unexpected street types to the appropriate ones in the expected list.
    You have to add mappings only for the actual problems you find in this OSMFILE,
    not a generalized solution, since that may and will depend on the particular area you are auditing.
- write the update_name function, to actually fix the street name.
    The function takes a string with street name as an argument and should return the fixed name
    We have provided a simple test so that you see what exactly is expected
"""
import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

OSMFILE = "san-diego_california.osm"
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)


expected = ["Street", "Avenue", "Boulevard", "Circle", "Calle", "Drive", "Court", "Caminito","Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Commons","Avenida", "Way", "Point", "Row", "Real", "Terrace", "Verde", "Vista", "Walk", "Highway"]
# Expected names in the dataset

mapping = { "St": "Street",
            "St.": "Street",
            "AVE": "Avenue",
            "Ave": "Avenue",
            "Ave.": "Avenue",
            "Av.": "Avenue",
            "Av": "Avenue",
            "ave": "Avenue",
            "Bl": "Boulevard",
            "Blvd": "Boulevard",
            "Blvd.": "Boulevard",
            "boulevard": "Boulevard",
            "CT": "Court",
            "Ct": "Court",
            "Dr": "Drive",
            "Dr.": "Drive",
            "E": "East",
            "Pk": "Park",
            "Pl": "Plaza",
            "Py": "Parkway",
            "Rd": "Road",
            "St": "street",
            }

# Search string for the regex. If it is matched and not in the expected list then add this as a key to the set.
def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)


def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")


def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    for event, elem in ET.iterparse(osm_file, events=("start",)):

        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
    osm_file.close()
    return street_types

# Cleaning the identified abbreviated street names to full names
def update_name(name, mapping):

# Updating certain specific incorrect street names as desired street names

    if name == '10511 Caminito Alvarez':
        return 'Caminito Alvarez'
    if 'Andada' in name:
        name = name.replace('Andada', '').strip()
    if '610 Paseo Del Rey Canyon Community Church' in name:
        name = name.replace('Paseo Del Rey Canyon', '').strip()
    if "College Grove Shopping Center W Of Sam'S Club Pkg Area" in name:
        name = name.replace('College Grove', '').strip()
    if name == 'South Bay Expressway CA-125':
        return 'South Bay Expressway'
    if name == 'Murphy Canyon Road #101':
        return 'Murphy Canyon Road'


    new_name = []

    for word in name.split(' '):
        if word in mapping:
            word = mapping[word]
        new_name.append(word)
    name = ' '.join(new_name)

    return name

def test():
    st_types = audit(OSMFILE)
    assert len(st_types) 
    pprint.pprint(dict(st_types))

    for st_type, ways in st_types.iteritems():
        for name in ways:
            better_name = update_name(name, mapping)
            print name, "=>", better_name
            if name == "El Cajon Bl":
                assert better_name == "El Cajon Boulevard"
            if name == "Cornerstone Ct":
                assert better_name == "Cornerstone Court"
            
if __name__ == '__main__':
    test()
    
# Auditing the post code 

def is_postcode(elem):
    return (elem.attrib['k'] == "addr:postcode")

def audit_postcode(postcodes, postcode):
    postcodes[postcode].add(postcode)
    return postcodes

def get_element(OSM_PATH, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(OSMFILE, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()

def audit(osmfile):
    osm_file = open(osmfile, "r")

    street_types = defaultdict(set)
    postcodes = defaultdict(set)
    for i, elem in enumerate(get_element(osmfile)):
    #for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            #modifies behavior of .iter and only return "tag" tags
            for tag in elem.iter("tag"):
                if is_postcode(tag):
                    postcodes = audit_postcode(postcodes, tag.attrib['v'])
    osm_file.close()
    pprint.pprint(dict(postcodes))

audit('san-diego_california.osm')


#Cleaning the identified errors in postcodes

test = ['92010-1407','92037-4291','92071-4417','92102-4810','CA 91914','92093-0068','92093-0094', '92101-3414, ','92101-6144','92111-2201']
print 'Cleaned Zipcodes:'
def update_postcode(postcode):
    # new regular expression pattern
    search = re.match(r'^\D*(\d{5}).*', postcode)
    # select the group that is captured
    clean_postcode = search.group(1)
    return clean_postcode
            
for item in test:
    cleaned = update_postcode(item)
    print cleaned