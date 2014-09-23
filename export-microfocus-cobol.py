#!/usr/bin/env python3
"""Script to parse and export data from Microfocus COBOL databases.

Requires you to provide a data definition file, because the database itself
does not contain the schema.

(c) 2010 by Michael Elsdörfer <michael@elsdoerfer.com>.
Published under MIT License.

Contains code from the script posted here:
  http://www.tek-tips.com/viewthread.cfm?qid=1510638&page=1

Follows the file format description available at:
  http://supportline.microfocus.com/documentation/books/sx20books/fhfile.htm
"""

import os, sys, string, array, struct
import datetime
import collections
import argparse
from collections import OrderedDict


# The functions zoned2num() and packed2num() are taken from Carey Evans
# (http://mail.python.org/pipermail/python-list/2000-April/665162.html)
def zoned2num(z):
    a = array.array('B', z)
    v = 0
    for i in a:
        v = (v * 10) + (i & 0xf)
    if (a[-1] & 0xf0) == 0xd0:
        v = -v
    return v

def packed2num(p):
    a = array.array('B', p)
    v = 0
    for i in a[:-1]:
        v = (v * 100) + (((i & 0xf0) >> 4) * 10) + (i & 0xf)
    i = a[-1]
    v = (v * 10) + ((i & 0xf0) >> 4)
    if (i & 0xf) == 0xd:
        v = -v
    return v


ENCODING = 'iso-8859-1'  # TODO: Should this be configurable? Probably.

def parse_config(config_file):
    """Returns list of fields definition from a config file.

    See example file for a definition of the format.

    Originally based on:
        http://www.tek-tips.com/viewthread.cfm?qid=1510638&page=1
    """
    Field = collections.namedtuple('Field', ['type', 'length', 'key', 'validator'])
    fields_by_index = []
    fields_by_key = {}

    nr_lines = 0
    for line in open(config_file):
        nr_lines += 1
        cfg_line = line
        # remove comments from line
        if '#' in cfg_line:
            cfg_line = cfg_line[:cfg_line.index('#')]
        # strip blanks from both ends
        cfg_line = cfg_line.strip()
        # process not empty lines
        if len(cfg_line) > 0:
            if cfg_line.startswith('validate'):
                key, expression = cfg_line[len('validate'):].split(':', 1)
                # Field to validate is given either by index, key, or its the
                # last known field.
                if key.isdigit():
                    target_index = int(key)
                elif not key:
                    target_index = len(fields_by_index) - 1
                else:
                    if not key in fields_by_key:
                        raise ValueError(
                        ("Error in config line %d: '%s' (References undefined " +
                         "key %s") % (nr_lines, cfg_line, key))
                    target_index = fields_by_key[key]

                fields_by_index[target_index] = \
                    fields_by_index[target_index]._replace(validator=expression.strip())
                continue

            # Repeat the last X lines with a different key prefix
            # Only already attached validators will be copied.
            if cfg_line.startswith('repeat'):
                last_x, new_prefix = cfg_line[len('repeat'):].split(':', 1)
                new_prefix = new_prefix.strip()
                last_x = int(last_x)
                for f in fields_by_index[-last_x:]:
                    nf = f._replace(key=new_prefix + f.key[len(new_prefix):] if f.key else '')
                    fields_by_index.append(nf)
                    if nf.key:
                        fields_by_key[field_key] = len(fields_by_index) - 1
                continue

            if ':' in cfg_line:
                # split into a list
                field_type, other_part = cfg_line.split(':', 1)
                field_type = field_type.upper()
                field_length, field_key = (other_part+' ').split(' ', 1)
                field_key = field_key.strip()

                if not field_length.isdigit():
                    raise ValueError(
                        ("Error in config line %d: '%s' (Data type " +
                         "length is not numeric)") % (nr_lines, cfg_line))

                field_length = int(field_length)
                # compute the length of Packed Decimal (COMP-3)
                if field_type == 'P':
                    field_length = field_length//2 + 1

                # Add the field to result
                field = Field(field_type, field_length, field_key, None)
                fields_by_index.append(field)
                if field_key:
                    fields_by_key[field_key] = len(fields_by_index) - 1

            else:
                raise ValueError(
                    ("Error in config line %d: '%s' (Line should " +
                     "have a form <DataType>:<length>)") % (nr_lines, cfg_line))

    return fields_by_index


class DataFileHeader(object):
    """Represents a Microfocus COBOL .DAT file header.
    """

    ORGANIZATION_SEQUENTIAL = 0x1
    ORGANIZATION_INDEXED = 0x2
    ORGANIZATION_RELATIVE = 0x3

    COMPRESSION_NONE = 0x0
    COMPRESSION_CBLDC001 = 0x1
    COMPRESSION_USER = 0x80

    RECORDING_FIXED = 0x0
    RECORDING_VARIABLE = 0x1

    INDEX_NONE = 0x0
    INDEX_FMT1 = 0x1
    INDEX_FMT2 = 0x2
    INDEX_FMT3 = 0x3
    INDEX_FMT4 = 0x4
    INDEX_FMT8 = 0x8

    # This is static
    size = 128

    def __init__(self):
        # Those two basically determine the file format. See the Microfocus
        # docs for the different combinations.
        self.organization = None
        self.recording_mode = None
        # If organization=index, this differentiates further.
        self.index_type = None
        # Whether the record header is 2 or 4 bytes.
        self.long_records = None
        # Minimum and maximum length of a record.
        self.max_record_length = None
        self.min_record_length = None
        # When the file was created (updated?)
        self.creation_date = None
        # The type of data compression used.
        self.data_compression = None
        # Database sequence number, used by add-on products.
        self.db_sequence_num = None
        # Version and build data for the indexed file handler creating
        # the file. Indexed files only.
        self.indexed_handler_version = None
        # Integrity flag. Indexed files only. If this is non-zero when
        # the header is read, it indicates that the file is corrupt.
        self.integrity_flag = None

    @property
    def organization_display(self):
        return {
            None: '(uninitialized)',
            self.ORGANIZATION_SEQUENTIAL: 'sequential',
            self.ORGANIZATION_INDEXED: 'indexed',
            self.ORGANIZATION_RELATIVE: 'relative',
        }.get(self.organization, '(unknown)')

    @property
    def recording_mode_display(self):
        return {
            None: '(uninitialized)',
            self.RECORDING_FIXED: 'fixed',
            self.RECORDING_VARIABLE: 'variable',
        }.get(self.recording_mode, '(unknown)')

    @property
    def data_compression_display(self):
        if self.data_compression is None:
            return '(uninitialized)'
        elif self.data_compression >= 0x2 and self.data_compression <= 0x7f:
            return '%d (reserved for internal use)' % self.data_compression
        elif self.data_compression >= self.COMPRESSION_USER:
            return '%d (user defined routine)' % self.data_compression
        else:
            return {
                self.COMPRESSION_NONE: 'none',
                self.COMPRESSION_CBLDC001: 'CBLDC001',
            }.get(self.data_compression, '(unknown)')

    @property
    def index_type_display(self):
        return {
            None: '(uninitialized)',
            self.INDEX_NONE: '(not set)',
            self.INDEX_FMT1: 'IDXFORMAT"1"',
            self.INDEX_FMT2: 'IDXFORMAT"2"',
            self.INDEX_FMT3: 'IDXFORMAT"3"',
            self.INDEX_FMT4: 'IDXFORMAT"4"',
            self.INDEX_FMT8: 'IDXFORMAT"8"',
        }.get(self.index_type, '(unknown)')

    @property
    def index_properties(self):
        """The type of index used can in some cases affect how data
        is stored on the disc. This may affect how we have to read the
        data.

        See the table "Physical Characteristics" in the Microfocus docs.
        """
        return {
            None: {},
            self.INDEX_NONE: {},
            self.INDEX_FMT1: {'record_alignment': 1, 'file_pointer_size': 4},
            self.INDEX_FMT2: {'record_alignment': 1, 'file_pointer_size': 4},
            self.INDEX_FMT3: {'record_alignment': 4, 'file_pointer_size': 4},
            self.INDEX_FMT4: {'record_alignment': 4, 'file_pointer_size': 4},
            self.INDEX_FMT8: {'record_alignment': 8, 'file_pointer_size': 6},
        }[self.index_type]

    @property
    def num_alignment_bytes(self):
        if (self.organization == DataFileHeader.ORGANIZATION_SEQUENTIAL and
            self.recording_mode == DataFileHeader.RECORDING_VARIABLE):
            return 4
        elif self.organization == DataFileHeader.ORGANIZATION_INDEXED:
            # I have indexed files where index type is 0, and they seem to
            # use a padding of 4.
            return self.index_properties.get('record_alignment', 4)
        else:
            return None

    @property
    def filetype_as_standardized_name(self):
        """The Microfocus documentation nicely explains the different types
        of files supported, as a combination of the "organization" and
        "recording mode" options. This returns the file type as a name
        as used in the docs.
        """
        try:
            organization = {
                    self.ORGANIZATION_INDEXED: 'Indexed File',
                    self.ORGANIZATION_SEQUENTIAL: 'Record Sequential File',
                    self.ORGANIZATION_RELATIVE: 'Relative File',
                }[self.organization]
            recording = {
                    self.RECORDING_FIXED: 'Fixed Length Records',
                    self.RECORDING_VARIABLE: 'Variable Length Records',
                }[self.recording_mode]
        except KeyError:
            return 'Unknown file type (probably not a Microfocus COBOL database)'
        else:
            return "{} with {}".format(organization, recording)

    def print(self):
        props_to_print = OrderedDict((
            ('Organization', 'organization_display'),
            ('Recording Mode', 'recording_mode_display'),
            ('Index Type', 'index_type_display'),
            ('Long Records', 'long_records'),
            ('Maximum Record Length', 'max_record_length'),
            ('Minimum Record Length', 'min_record_length'),
            ('Creation Date', 'creation_date'),
            ('Data Compression', 'data_compression_display'),
            ('DB Sequence Number', 'db_sequence_num'),
            ('Indexed Handler Version', 'indexed_handler_version'),
            ('Integrity Flag', 'integrity_flag'),
        ))

        max_prop_len = max([len(p) for p in props_to_print.keys()])
        for display, prop in props_to_print.items():
            print('{:>{width}}: {}'.format(
                display, getattr(self, prop), width=max_prop_len+3))


class DataFileRecord(object):
    """Represents a Microfocus COBOL .DAT file record.
    """

    # Not an actual type defined by the docs, but I have at least one
    # file where there seems to be a SYSTEM-like record, but with a record
    # type of zero, as well as no length. Defining a null record is the
    # most straightforward way to deal with this, as the parsing code will
    # then just continue to yield null records until we find a byte that
    # starts another one.
    # It's possible that the data free space records from the index, which
    # we're not parsing, would indicate these null bytes as free space.
    TYPE_NULL = 0b0000
    # Seemingly used for "data free space records" in "variable format
    # indexed files"; in fixed format indexed files, these records are
    # in the index file itself.
    TYPE_SYSTEM = 0b001
    # Deleted record (available for reuse via the free space list).
    TYPE_DELETED = 0b010
    # This is used for the file header
    TYPE_HEADER = 0b0011
    # Normal user data record.
    TYPE_NORMAL = 0b0100
    # Reduced user data record (indexed files only).
    TYPE_REDUCED = 0b0101
    # Pointer record (indexed files only).
    TYPE_POINTER = 0b0110
    # User data record referenced by a pointer record.
    TYPE_REFERENCED = 0b0111
    # Reduced user data record referenced by a pointer record.
    TYPE_REDUCED_REFERENCED = 0b1000

    def __init__(self, type, bytes):
        self.type = type
        self.bytes = bytes

    @property
    def type_display(self):
        return {
            None: '(uninitialized)',
            self.TYPE_NULL: '(null)',
            self.TYPE_SYSTEM: 'system',
            self.TYPE_DELETED: 'deleted',
            self.TYPE_HEADER: 'header',
            self.TYPE_NORMAL: 'normal',
            self.TYPE_REDUCED: 'reduced',
            self.TYPE_POINTER: 'pointer',
            self.TYPE_REFERENCED: 'referenced',
            self.TYPE_REDUCED_REFERENCED: 'reduced_referenced',
        }.get(self.type, '(unknown)')


def print_field_match_debug(matches):
    """Matches is a list of tuples representing how the bytes of a
    record have been assigned to the fields in a definition.

        (start_byte, end_byte, bytes, field)

    This will print this data in readable way using colors.
    """
    from clint.textui.cols import console_width
    from clint.packages.colorama import Back
    COLORS = ('RED', 'BLUE')
    width = console_width({})
    col = 0
    key_subline = ''
    for idx, (start, end, bytes, field) in enumerate(matches):
        if col + len(bytes) > width:
            col = 0
            sys.stdout.write('\n')
            sys.stdout.write(key_subline + '\n')
            key_subline = ''

        sys.stdout.write(getattr(Back, COLORS[idx % len(COLORS)]))
        sys.stdout.write(bytes.decode('latin1'))
        sys.stdout.write(Back.RESET)
        col += len(bytes)

        key_subline += "%s" % (field.key or '')[:len(bytes)].ljust(len(bytes))

    sys.stdout.write('\n')
    sys.stdout.write(key_subline + '\n')


def parse_record_fields(record_bytes, field_def):
    """Take the raw bytes of a record, apply the field definition.

    Return a 2-tuple of (list, dict). The list contains all non-skipped
    fields. The dict contains all non-skipped fields that have been
    assigned a key.
    """
    start_byte = 0
    end_byte = 0
    total_bytes = 0
    nr_fld = 0
    parsed_list = []
    parsed_map = {}
    matches_with_pos = []
    for field in field_def:
        nr_fld += 1
        total_bytes += field[1]
        end_byte = total_bytes
        field_bytes = record_bytes[start_byte:end_byte]
        skipped = False

        # For debugging output keep a list of data pieces and the fields
        # that they matched.
        matches_with_pos.append((start_byte, end_byte, field_bytes, field))

        # Parse the field data based on type
        if field[0] == '*':
            fld_data = field_bytes
            skipped = True
        elif field[0] == 'X':
            fld_data = field_bytes.decode(ENCODING).rstrip()
        elif field[0] == 'Z':
            fld_data_num = zoned2num(field_bytes)
            fld_data = str(fld_data_num).strip()
        elif field[0] == 'P':
            fld_data_num = packed2num(field_bytes)
            fld_data = str(fld_data_num).strip()
            format_packed = True
            if format_packed:
                if fld_data_num == 0:
                    fld_data = '0.00'
                else:
                    fld_data = fld_data[:-2] + '.' + fld_data[-2:]
        else:
            raise ValueError('Unknown field type: %s' % repr(field[0]))

        # Run the field validator
        if field.validator:
            try:
                if not eval(field.validator, {
                    'v': fld_data, 'fields_map': parsed_map}):
                    raise ValueError('validator did not return True')
            except Exception as e:
                print_field_match_debug(matches_with_pos)
                raise RuntimeError(('Field %s (key %s) in record has data "%s" and '
                    'fails validator (%s): %s\n\n%s' % (
                    nr_fld, field.key, fld_data, field.validator, e, record_bytes
                )))

        # Add to result set
        if not skipped:
            parsed_list.append(fld_data)
            if field.key:
                parsed_map[field.key] = fld_data
        start_byte = end_byte

    if end_byte < len(record_bytes):
        raise ValueError("%s unread bytes at end of record" % (len(record_bytes) - end_byte))

    return parsed_list, parsed_map


class CobolDataFile(object):
    """Parse the data from ``data_file`` applying the field defintions in
    ``field_list``.
    """

    def __init__(self, fileobj):
        self.file = fileobj
        self.warnings = []

        self._read_header()

    def read(self, num_bytes, no_warn=False):
        """Read helper that does validation.

        ``no_warn`` disables the warning if no bytes at all can be read. This
        is used for when we expect EOF.
        """
        data = self.file.read(num_bytes)
        if len(data) != num_bytes and not (no_warn and len(data) == 0):
            raise IOError(('Wanted to read {} bytes at offset {}, got only '
                           '{} before end of file').format(
                               num_bytes, self.file.tell()-len(data), len(data)))
        return data

    def _warn(self, msg, *a, **kw):
        self.warnings.append(msg.format(*a, **kw))

    def _read_header(self):
        self.header = header = DataFileHeader()

        def read(num_bytes, name=None, unpack=None, match=None, transform=None):
            """Read ``num_bytes`` from file. Apply ``unpack``.
            Apply ``transform``. Match it against ``match``. Store it in
            ``name`` attribute of the header. In this order.
            """
            try:
                data = raw_data = self.read(num_bytes)
            except IOError as e:
                self._warn("%s" % e)
                return
            if len(raw_data) != num_bytes:
                self._warn('Wanted to read {} bytes at offset {}, got only '
                           '{} before end of file',
                           num_bytes, self.file.tell()-num_bytes, len(raw_data))
            if unpack:
                data = struct.unpack(unpack, data)[0]
            if transform:
                data = transform(data)
            if match and not data in match:
                self._warn('Got {} at offset {}, expected{modifier}: {}',
                           raw_data, self.file.tell()-num_bytes,
                           ", ".join(map(str, match)),
                           modifier=' one of' if len(match)>1 else '')
            if name:
                if isinstance(match, dict):
                    data = match.get(data, data)
                setattr(header, name, data)
            return data

        def parse_date(data):
            # The last two characters are defined as "CC" / "cents" in
            # COBOL; it's probably one hundred's of a second, but we can't
            # parse that in a Python dateformat.
            # Also, the encoding used here might not be correct.
            try:
                return datetime.datetime.strptime(
                    data[:-2].decode(ENCODING), '%y%m%d%H%M%S')
            except ValueError as e:
                self._warn('Record has an invalid date value: {}; error was: {}',
                           data, e)
                return data

        # 0
        read(4, 'long_records', match={
            # These bytes have meaning, but in the end, there are
            # only two possible versions.
            b'\x30\x7E\x00\x00': False,
            b'\x30\x00\x00\x7C': True,
        })
        # 4
        read(2, 'db_sequence_num', '>h')
        # 6
        read(2, 'integrity_flag', '>h')
        # 8
        read(14, 'creation_date', transform=parse_date)
        # 22, reserved
        read(14)
        # 36, reserved
        read(2, match=(b'\x00\x3e',))
        # 38, not used, set to zeros
        read(1, match=(b'\x00',))
        # 39
        organization = read(1, 'organization', '>b', match=(1,2,3))
        # 40
        read(1, match=(b'\x00',))
        # 41
        read(1, 'data_compression', '>b')
        # 42
        read(1, match=(b'\x00',))
        # 43
        read(1, 'index_type', '>b')
        # 44
        read(4)
        # 48
        read(1, 'recording_mode', '>b', match=(0,1))
        # 49
        read(5, match=(b'\x00'*5,))
        # 54
        read(4, 'max_record_length', '>i')
        # 58
        read(4, 'min_record_length', '>i')
        if organization == DataFileHeader.ORGANIZATION_INDEXED:
            # 62, set to zero
            read(14, match=(b'\x00'*14,))
            # 76, reserved
            read(1, match=(b'\x04',))
            # 77, set to zero
            read(31, match=(b'\x00'*31,))
        else:
            # 62, set to zero
            read(46, match=(b'\x00'*46,))
        # 108
        read(4, 'indexed_handler_version', '>i')
        if organization == DataFileHeader.ORGANIZATION_INDEXED:
            # 112
            read(8, match=(b'\x00'*8,))
            # 120
            read(8, 'logical_end_offset', '>q')
        else:
            # 112
            read(16, match=(b'\x00'*16,))

    def validate(self):
        """Validate the header information.
        """
        if not getattr(self, '_validated', False):
            if self.header.integrity_flag != 0:
                self._warn('Integrity flag is non-zero, this may indicate a '
                           'corrupt file!')
            if self.header.data_compression != DataFileHeader.COMPRESSION_NONE:
                self._warn('The file indicates it uses compression, we '
                           'probably won\'t handle this properly.')
            # There are other combinations of organization and recording mode
            # which we also don't support, but those don't even have a header
            # in the first place.
            if self.header.organization == DataFileHeader.ORGANIZATION_RELATIVE:
                self._warn('This seems to be a relative file, which are not '
                           'supported. This probably won\'t be handled '
                           'correctly. Feel free to submit a patch.')

            self._validated = True

    def iter_records(self, ignore=(DataFileRecord.TYPE_HEADER,
                                   DataFileRecord.TYPE_NULL,
                                   DataFileRecord.TYPE_SYSTEM)):
        self.validate()
        header = self.header

        self.file.seek(header.size)   # go to data
        while True:
            # Read the record header
            header_length = 4 if header.long_records else 2
            record_header_bytes = self.read(header_length, no_warn=True)
            if not record_header_bytes:
                return  # We're done

            # Parse the record header; this can be 2 or 4 bytes, depending
            # on the maximum record length, as defined in the header.
            record_header = struct.unpack(
                '>i' if header.long_records else '>h',
                record_header_bytes)[0]
            # First 4 bits are the record type
            record_type = (record_header >> (27 if header.long_records else 12))
            if record_type > 8 or record_type < 0:
                self._warn('Unknown record type id in record header at '+
                           'offset {}: {}; This is probably a corrupt file.',
                           self.file.tell()-header_length, record_type)
            # Next bytes indicate the length of the record; simply zero
            # out the first nibble to get the length.
            if header.long_records:
                data_length = (record_header & 0x0FFFFFFF)
            else:
                data_length = (record_header & 0x0FFF)

            # Get the actual record
            record_data = self.read(data_length)

            # Parse the record
            if not record_type in ignore:
                yield DataFileRecord(record_type, record_data)

            # Read the slack bytes. The formula is:
            #    number_slack_bytes = record_length % num_alignment_bytes
            # with record_length being the actual length of the record (
            # meaning the header plus the data length as specified by the
            # header), and num_alignment_bytes being dependent on the file
            # type.
            if header.num_alignment_bytes:
                self.read(
                    -(data_length + header_length) % header.num_alignment_bytes)


def parse_records(records_iter, field_def):
    """Records in, (record, parsed) out."""
    for record in records_iter:
        # TODO: what about pointers, references?
        if record.type in (
                DataFileRecord.TYPE_NORMAL, DataFileRecord.TYPE_DELETED):
            if not field_def is None:
                yield record, parse_record_fields(record.bytes, field_def)
            else:
                yield record, None
        else:
            raise ValueError('Unexpected record type: %s' % record.type_display)


def csv_exporter(records, output):
    for index, (record, data) in enumerate(records):
        output.write(",".join(data)+'\n')

def json_exporter(records, output):
    import json
    result = []
    for index, (record, data) in enumerate(records):
        # If list and map contain the same items, output only map, otherwise both
        if len(data[0]) == len(data[1].keys()):
            result.append(data[1])
        else:
            result.append(data)
    output.write(json.dumps(result, indent=4))

def bytes_exporter(records, output):
    for index, (record, _) in enumerate(records):
        print('')
        print("%s: %s" % (record.type, repr(record.bytes)[1:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fields', type=str,
        help='Use the given field definition to export the data')
    parser.add_argument('--bytes', action="store_true",
        help='Output raw bytes of every record in database')
    parser.add_argument('--csv', action="store_true",
        help='Output database in CSV format')
    parser.add_argument('--json', action="store_true",
        help='Output database in JSON format')
    parser.add_argument('files', metavar='DATAFILE', nargs='+')
    args = parser.parse_args()

    # Load the field definition file
    fields = None
    if args.fields:
        fields = parse_config(args.fields)

    # Are we asked to output the database records?
    exporter = None
    if args.bytes:
        exporter = bytes_exporter
    elif args.json:
        exporter = json_exporter
    elif args.csv or fields:  # Use as the default
        exporter = csv_exporter

    if exporter:
        if len(args.files) != 1:
            print('Only a single database file is supported when exporting.')
            sys.exit(1)

        for filename in args.files:
            file = CobolDataFile(open(filename, 'rb'))
            exporter(parse_records(file.iter_records(), fields), sys.stdout)

    # Only output the header
    else:
        for filename in args.files:
            file = CobolDataFile(open(filename, 'rb'))
            print("{}: {}".format(filename, file.header.filetype_as_standardized_name))
            file.header.print()

            sum = len(list(file.iter_records()))
            print('')
            print('The file contains {} records.'.format(sum))

    # In both cases, the file may have warnings at the end
    if file.warnings:
        print('!!! THERE ARE WARNINGS !!!')
        for w in file.warnings:
            print('  {}'.format(w))
        print()
