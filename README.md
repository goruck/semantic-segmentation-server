*Work in progress - p/o the radar-ml project*

# semantic-segmentation-server

## Semantic segmentation served over grpc using Google Edge TPU.

```protobuf
service SemanticSegmentation {
    // A simple RPC. 
    rpc GetSegmentedObjects(Empty) returns (SegmentedObjectData) {}
    rpc GetCameraResolution(Empty) returns (CameraResolution) {}
  }

message SegmentedObject {
    // Most likely semantic segment label.
    string label = 1;

    // Score of label.
    // This can be used as a measure of confidence.
    float score = 2;

    // Relative area of segment. Max = 1.
    float area = 3;

    // Relative segment centroid coords.
    // (0,0) is top left of image containing the segment.
    // (1,0) is top right "".
    // (0,1) is bottom left "".
    // (1,1) is bottom right "".
    message Centroid {
        float cx = 1;
        float cy = 2;
    }

    Centroid centroid = 4;
}

message SegmentedObjectData {
    // Recognized and segmented object data.
    repeated SegmentedObject data = 1;
}

message CameraResolution {
    int32 width = 1;
    int32 height = 2;
}

message Empty {
    // Placeholder.
}
```