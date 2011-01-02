This is a script to access data in a Microfocus COBOL database and export
it. It runs on Python 3 and requires the argparse library (needs to be
installed on Python < 3.2).

I'm not at all familiar with COBOL, and built this based on information I
was able to gather on the web, and looking at hex dumps. It's not a proper
library to work with these files, but a cludge to get the data out. It
works for the files I need to process, and might need furter improvements
for yours.

COBOL seems to have dealing with these types of databases built into the
language itself. Unfortuately, the files itself do not contain a schema;
the schema is written in COBOL code, in copybooks or FDD files.

As a result, to actually export the data, we need to define the fields
that it contains. Lacking the original source code or definition files,
this needs to be reverse-engeneered. This exporter uses an ad-hoc format
to define the fields, rather than actual COBOL .FDD files.


Usage
-----

Print the headers::

    $ ./mfcobol-export FILE1.DAT
	 FILE1.DAT: Indexed File with Variable Length Records
              Organization: index
            Recording Mode: variable
                Index Type: (not set)
              Long Records: False
     Maximum Record Length: 155
     Minimum Record Length: 155
             Creation Date: 1996-09-06 09:29:31
          Data Compression: none
        DB Sequence Number: 0
   Indexed Handler Version: 1090527224
            Integrity Flag: 0


Export the data::

	 $ ./mfcobol-export --export-with=datadef.cfg FILE1.DAT


For more information on how to declare fields in the .cfg file, see the
example file that is included.


Supported file types
--------------------

Microfocus COBOL supports different types of databases:

    1. Line Sequential (no header)
	 2. Printer Sequential (no header)
	 3. Record Sequential Files with Fixed Length Records (no header)
	 4. Record Sequential Files with Variable Length Records
    5. Relative Files with Fixed Length Records (no header)
	 6. Relative Files with Variable Length Records
	 7. Indexed Files

Of those, files containing no header are generally not supported by
this script. In addition, "Relative Files with Variable Length Records"
are also not supported. This leaves 4. and 7.

Personally tested I have only indexed files (7).
Variable Length Record Sequential should work as well, though, since it's
similar.


Resources used
--------------

http://supportline.microfocus.com/documentation/books/nx40/fhorgs.htm
    Official documentation on the file format.

http://www.tek-tips.com/viewthread.cfm?qid=1510638&page=1
	 Script to parse database records via ad-hoc field definitions. Used
	 by this exporter.

http://g4u0419c.houston.hp.com/en/64/hppg/hpmixc.htm
	 Explains the datatypes of Microfocus COBOL.


Other related tools
-------------------

http://cb2java.sourceforge.net/
    Requires the COBOL copybook files, then can probably export the data.

http://www.cobolproducts.com/
    Silber Systems DataViewer and related products; Very expensive, not a
	 great UI. Can guess the record layout, but this doesn't really work
	 particularily well.

http://www.alchemysolutions.com/products/Fujitsu-Data-Converter-for-Windows/overview
    Fujitsu Data Converter for Windows, part of their NetCOBOL product. No
	 demo version, prices only on request.

http://www.connx.com/products/microfocus.html (
    CONNX for Microfocus
