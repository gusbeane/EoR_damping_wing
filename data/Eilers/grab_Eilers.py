import os
from astropy import units as u

from specdb import query_catalog as spqcat
from specdb import interface_db as spgidb
from specdb import utils as spdbu
from specdb.specdb import SpecDB, IgmSpec
from specdb import specdb as sdbsdb

igmsp = sdbsdb.IgmSpec()
