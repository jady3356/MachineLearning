'use strict';

const express = require('express');
const router = express.Router();

router.get('/', require('./welcome'));

router.get('/getPoetry0', require('./getPoetry0'));
router.get('/getPoetry0_split', require('./getPoetry0_split'));
router.get('/getPoetry1', require('./getPoetry1'));
router.get('/getPoetry2', require('./getPoetry2'));
router.get('/getPoetry3', require('./getPoetry3'));
router.get('/getPoetry4', require('./getPoetry4'));
router.get('/getPoetry5', require('./getPoetry5'));
router.get('/getPoetry6', require('./getPoetry6'));
router.get('/getPoetry7', require('./getPoetry7'));
router.get('/getPoetry8', require('./getPoetry8'));
router.get('/getPoetry9', require('./getPoetry9'));

router.get('/login', require('./login'));
router.get('/user', require('./user'));
router.all('/tunnel', require('./tunnel'));

module.exports = router;