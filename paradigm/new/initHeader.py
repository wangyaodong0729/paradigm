import struct


def init_header(chanID, code, request, samples, sizeBody, sizeUn):
    # Convert input arguments to byte arrays with swapped byte order
    c_chID = struct.pack('>4s', chanID.encode('utf-8'))
    w_Code = struct.pack('>H', code)
    w_Request = struct.pack('>H', request)
    un_Sample = struct.pack('>I', samples)
    un_Size = struct.pack('>I', sizeBody)
    un_SizeUn = struct.pack('>I', sizeUn)

    # Concatenate byte arrays to form the header
    header = c_chID + w_Code + w_Request + un_Sample + un_Size + un_SizeUn

    return header


def control_code(par_type):
    if par_type == 'CTRL_FromServer':
        return 1
    elif par_type == 'CTRL_FromClient':
        return 2
    else:
        return -1


def request_type(par_type):
    if par_type == 'RequestVersion':
        return 1
    elif par_type == 'RequestChannelInfo':
        return 3
    elif par_type == 'RequestBasicInfoAcq':
        return 6
    elif par_type == 'RequestStreamingStart':
        return 8
    elif par_type == 'RequestStreamingStop':
        return 9
    else:
        return -1


header = init_header('CTRL', control_code('CTRL_FromClient'), request_type('RequestStreamingStart'),0,0,0)
stop_header = init_header('CTRL', control_code('CTRL_FromClient'), request_type('RequestStreamingStop'),0,0,0)
print(header)
print(stop_header)