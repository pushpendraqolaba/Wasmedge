// Copyright Joyent and Node contributors. All rights reserved. MIT license.
'use strict';
import common from '../common';
import fs from 'fs';
import path from 'path';
import assert from 'assert';
import tmpdir from '../common/tmpdir';
const file = path.join(tmpdir.path, 'write_stream_filehandle_test.txt');
const input = 'hello world';

tmpdir.refresh();

fs.promises.open(file, 'w+').then((handle) => {
  handle.on('close', common.mustCall());
  const stream = fs.createWriteStream(null, { fd: handle });

  stream.end(input);
  stream.on('close', common.mustCall(() => {
    const output = fs.readFileSync(file, 'utf-8');
    assert.strictEqual(output, input);
  }));
}).then(common.mustCall());
