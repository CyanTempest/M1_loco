#!/usr/bin/env python3
import json
from m1_locomotion.audit import audit_model

if __name__ == "__main__":
    print(json.dumps(audit_model(), indent=2))
