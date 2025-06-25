<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Scripts extends Model
{
      protected $connection = 'mysql';

    // ✅ Make sure this matches your MySQL table name
    protected $table = 'scripts';

    // ✅ List the fields you want to allow for mass assignment
    protected $fillable = [
        'script_id',
        'topic',
        'raw_script',
        'parsed_script',
        'created_at',
        'updated_at',
    ];

    // ✅ Disable default timestamps if you're manually managing them
    public $timestamps = false;
}
