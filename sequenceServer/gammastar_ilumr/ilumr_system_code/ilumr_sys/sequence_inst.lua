local ffi = require("ffi")

ffi.cdef[[
typedef enum {
	INIT = 0,
	END = 1,
	DELAY = 2,
	PULSESTART = 3,
	PULSEUPDATE = 4,
	PULSESTOP = 5,
	ACQUIRE = 6,
	SHIM = 7,
	GRADIENT = 8,
	WAIT_FOR_TRIGGER = 13,
	GPO_SET = 14,
	GPO_CLEAR = 15,
	PULSEUNBLANK = 16,
	PULSEBLANK = 17
} cmd_t;

typedef struct __attribute__ ((packed)) {
	cmd_t cmd; // must be first in every wiredata struct
	int amp_enable;
	int rx_gain;
} init;

typedef struct __attribute__ ((packed)) {
	cmd_t cmd;
} end;

typedef struct __attribute__ ((packed)) {
	cmd_t cmd;
	double time;
} delay;

typedef struct __attribute__ ((packed)) {
	cmd_t cmd;
	double freq;
	float phase;
	float amp;
} pulse_start;

typedef struct __attribute__ ((packed)) {
	cmd_t cmd;
	double freq;
	float phase;
	float amp;
} pulse_update;

typedef struct __attribute__ ((packed)) {
	cmd_t cmd;
} pulse_stop;

typedef struct __attribute__ ((packed)) {
	cmd_t cmd;
} pulse_unblank;

typedef struct __attribute__ ((packed)) {
	cmd_t cmd;
} pulse_blank;

typedef struct __attribute__ ((packed)) {
	cmd_t cmd;
	double freq;
	float phase;
	float dw;
	unsigned int samples;
} acquire;

typedef struct __attribute__ ((packed)) {
	cmd_t cmd;
	float shim[8]; // order: x, y, z, z2, zx, zy, xy, x2y2
} shim;

typedef struct __attribute__ ((packed)) {
	cmd_t cmd;
	float grad[3]; // order: x, y, z
} gradient;

typedef struct __attribute__ ((packed)) {
	cmd_t cmd;
	unsigned int mask;
} gpo_set;

typedef struct __attribute__ ((packed)) {
	cmd_t cmd;
	unsigned int mask;
} gpo_clear;
]]
