"""
Example client for sem_seg_server.py.

Copyright (c) 2020 Lindo St. Angel
"""

import grpc
import sem_seg_server_pb2
import sem_seg_server_pb2_grpc

def get_camera_resolution(stub):
    request = sem_seg_server_pb2.Empty()
    try:
        response = stub.GetCameraResolution(request)
        print('Camera resolution fetched.')
        return response
    except grpc.RpcError as err:
        print(err.details()) #pylint: disable=no-member
        print('{}, {}'.format(err.code().name, err.code().value)) #pylint: disable=no-member
        exit(1)

def get_detected_objects(stub):
    request = sem_seg_server_pb2.Empty()
    try:
        response = stub.GetSegmentedObjects(request)
        print('Detected object(s) fetched.')
        return response
    except grpc.RpcError as err:
        print(err.details()) #pylint: disable=no-member
        print('{}, {}'.format(err.code().name, err.code().value)) #pylint: disable=no-member
        exit(1)

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = sem_seg_server_pb2_grpc.SemanticSegmentationStub(channel)
        while True:
            res = get_camera_resolution(stub)
            print(res)
            res = get_detected_objects(stub)
            print(res)

if __name__ == '__main__':
    run()